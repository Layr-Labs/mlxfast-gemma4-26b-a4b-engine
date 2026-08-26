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

public func runDFlashGreedyRound(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    targetCache: inout [KVCache],
    draftCache: [KVCache],
    bonus: Int,
    targetHidden: MLXArray,
    promptTokenCount: Int,
    generatedTokenCount: Int,
    blockSize: Int,
    maxEmitCount: Int
) throws -> DFlashGreedyRoundResult {
    guard blockSize >= 2 else {
        throw DFlashError.invalidBlockSize(blockSize)
    }

    let rollbackProvider = target as? any DFlashTargetCacheRollbackProvider
    let targetRollbackState =
        rollbackProvider?.makeDFlashCacheRollbackState(cache: targetCache)
        ?? target.makeDefaultDFlashCacheRollbackState(cache: targetCache)

    let draftTokens = try drafter.draftBlock(
        bonus: bonus,
        targetHidden: targetHidden,
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
    let verifyOut: DFlashGreedyTargetForward =
        try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsEnabled {
            try target.forwardGreedyTokensForDFlash(
                verifyInput,
                cache: targetCache,
                targetLayerIds: drafter.config.targetLayerIds
            )
        }
    let targetTokens = verifyOut.tokens
    let verifiedTokenCount = targetTokens.dim(1)
    let draftTokenIds = draftTokens.squeezed(axis: 0)
    let targetTokenIds = targetTokens.squeezed(axis: 0)
    let proposedCount = Swift.max(0, blockSize - 1)
    let comparableCount = Swift.min(proposedCount, verifiedTokenCount)
    let walkedAccepted: Int
    let emitted: [Int]
    let accepted: Int
    if dFlashCPUAcceptWalk {
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
            eval(targetTokens, draftTokens, acceptedArray)
        } else {
            eval(targetTokens, draftTokens)
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
    let nextTargetHidden: MLXArray
    if let rollbackProvider {
        nextTargetHidden = try rollbackProvider.rollbackDFlashCache(
            &targetCache,
            state: targetRollbackState,
            verifyInput: verifyInput,
            acceptedTokenCount: accepted,
            rejectedTokenCount: trim,
            targetLayerIds: drafter.config.targetLayerIds,
            verifiedTargetHidden: verifyOut.targetHidden
        )
    } else {
        nextTargetHidden = try target.rollbackDFlashCacheUsingDefault(
            &targetCache,
            state: targetRollbackState,
            verifyInput: verifyInput,
            acceptedTokenCount: accepted,
            rejectedTokenCount: trim,
            targetLayerIds: drafter.config.targetLayerIds,
            verifiedTargetHidden: verifyOut.targetHidden
        )
    }

    return DFlashGreedyRoundResult(
        accepted: accepted,
        tokens: emitted,
        bonus: emitted.last ?? bonus,
        targetHidden: nextTargetHidden
    )
}
