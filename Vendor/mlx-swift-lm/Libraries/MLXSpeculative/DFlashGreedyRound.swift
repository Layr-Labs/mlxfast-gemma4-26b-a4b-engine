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
//
// ENGINE ADDITION (2026-09-02), also not in the fork: the optional
// `proposal:` argument documented on `runDFlashGreedyRound` below. It swaps
// the DRAFT SOURCE and nothing else — the verify input, the accept walk, the
// emit clamp and the rollback are the same lines running on the same
// rectangle.

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

/// Run one greedy DFlash round.
///
/// `proposal` is the ALTERNATE DRAFT SOURCE (engine addition, 2026-09-02, not
/// in the fork). When it is nil the round is exactly the ported one: the
/// neural drafter proposes `blockSize - 1` tokens. When it is non-nil those
/// tokens ARE the block, `drafter.draftBlock` is not called, and everything
/// downstream — the verify input, the rectangular target forward, the accept
/// walk, the emit clamp and the KV rollback — is byte-for-byte the same code.
/// A proposal is therefore not a fidelity change of any kind: every emitted
/// token is still the TARGET's own greedy argmax at that position, and a
/// wrong proposal costs exactly what a wrong drafter block costs, the
/// rejected tail of one round.
///
/// DRAFTER-CACHE ALIGNMENT, the one thing a caller must get right. The
/// drafter's KV cache does not hold the block it proposes — look at
/// `DFlashAttention.callAsFunction`: it caches keys/values of the CONTEXT
/// (the projected `targetHidden`) and merely concatenates the proposal's own
/// keys for that one forward. So one `draftBlock` call advances the draft
/// cache by `targetHidden.dim(1)` — the number of tokens the previous round
/// COMMITTED — and never by `blockSize`. That is what makes the trim below a
/// no-op in steady state: after `draftBlock`, `draftCache.offset` already
/// equals `promptTokenCount + generatedTokenCount - 1`.
///
/// A round that skips `draftBlock` therefore leaves the draft cache SHORT by
/// this round's committed tokens — the trim can only remove context, so it
/// cannot repair that, and it correctly does nothing (the delta is negative).
/// The caller closes the gap instead, and the seam for it is already in this
/// signature: `targetHidden` is a whole `[1, L, targetHidden]` context, not a
/// single row. A caller that skips the drafter for some rounds must ACCUMULATE
/// each skipped round's returned `targetHidden` and pass the concatenation on
/// the next drafter round. The drafter then caches exactly the same context
/// vectors, at exactly the same RoPE positions, that a run of ordinary rounds
/// would have cached — `contextKeys` is roped at `cache.offset`, which is
/// where those tokens actually live — so the cache is not "resynchronised
/// approximately", it is identical, and the bonus column stays consistent
/// because `bonus` is the last committed token either way.
/// `RuntimeWorkerDFlashFreeRunSession` is the reference caller.
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
    maxEmitCount: Int,
    proposal: [Int]? = nil
) throws -> DFlashGreedyRoundResult {
    guard blockSize >= 2 else {
        throw DFlashError.invalidBlockSize(blockSize)
    }

    let rollbackProvider = target as? any DFlashTargetCacheRollbackProvider
    let targetRollbackState =
        rollbackProvider?.makeDFlashCacheRollbackState(cache: targetCache)
        ?? target.makeDefaultDFlashCacheRollbackState(cache: targetCache)

    let draftTokens: MLXArray
    if let proposal {
        // The block the caller supplied IS the draft. Refuse a length that
        // disagrees with `blockSize` rather than silently verifying a
        // different rectangle than the caller budgeted for: every counter
        // downstream (`proposedCount`, `maxEmitCount`) is derived from
        // `blockSize`. The reported size is the block the proposal implies,
        // which is the honest half of the mismatch.
        guard proposal.count == blockSize - 1 else {
            throw DFlashError.invalidBlockSize(proposal.count + 1)
        }
        draftTokens = MLXArray(proposal.map(Int32.init))[.newAxis, .ellipsis]
    } else {
        draftTokens = try drafter.draftBlock(
            bonus: bonus,
            targetHidden: targetHidden,
            cache: draftCache,
            blockSize: blockSize
        )
    }
    asyncEval(draftTokens)

    // Steady state this is a no-op (see the alignment note above): a
    // `draftBlock` call has just advanced the draft cache to exactly this
    // frontier. It bites only when the drafter ran ahead of the committed
    // chain. On a `proposal` round the delta is NEGATIVE — the cache is
    // behind, not ahead — and the guard below correctly declines to act,
    // because trimming can only remove context; the caller re-feeds the
    // missing context on its next drafter round.
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
