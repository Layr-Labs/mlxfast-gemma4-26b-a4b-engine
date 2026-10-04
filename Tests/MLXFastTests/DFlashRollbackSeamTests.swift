import Foundation
import MLXLLM
import MLXLMCommon
import Testing

// The ROTATING-CACHE RING SEAM in the DFlash rollback, and the two ledgers a
// single-stream DFlash free run keeps against it.
//
// WHAT FAILED (receipt: .benchmark-artifacts/gemma4-humaneval/
// dflash-cycle-d3-164-receipt.json, 2026-09-02). Three of 164 HumanEval
// problems died with `untrimmableCache` — /94, /123, /129 — while 161 passed,
// including eight runs longer than the three that failed. Total length did not
// separate them. What separated them was the ROUND that first pushed the
// target's `RotatingKVCache` past its 1024-position ring: in all eight
// survivors that round committed its whole 2-wide block (nothing to roll back),
// and in the three failures it rejected.
//
// WHY. `makeDefaultDFlashCacheRollbackState` decided whether to snapshot from
// the offset the round STARTED at, and `rollbackDFlashCacheUsingDefault` asked
// `canTrimPromptCache` again after the verify had written the whole
// `[1, blockSize]` rectangle. `RotatingKVCache.isTrimmable` is
// `offset < maxSize`, so the one round that starts inside the ring and ends
// past it answered "trimmable" to the first question and "untrimmable" to the
// second: no snapshot, no trim, and the run failed on its first rejected token.
//
// These tests are unconditional and touch NO MLX. `RotatingKVCache` and
// `KVCacheSimple` allocate no array until something calls `update`, and
// `offset`, `isTrimmable`, `trim`, `copy` and `maxSize` are pure integer state,
// so the seam arithmetic is exercised against the REAL cache types and the REAL
// `trimPromptCache` on a host with no Metal.
//
// WHAT THEY DO NOT PROVE. Nothing here runs a model. That the target verify
// writes exactly `blockSize` positions, and that `DFlashDraftModel.draftBlock`
// advances the draft cache by exactly the context length it is handed, are
// facts about the MLX forwards, pinned by the box-only
// `Gemma4DFlashForwardTests`. `ScriptedRun` below encodes both as its advance
// rules and then checks that the BOOKKEEPING built on them is consistent — the
// half that was wrong.

@Suite("DFlash rollback ring seam")
struct DFlashRollbackSeamTests {

    // MARK: - The predicate

    @Test
    func aRoundThatStaysInsideTheRingNeedsNoSnapshot() {
        let cache: [KVCache] = [RotatingKVCache(maxSize: 1024, keep: 0)]
        cache[0].asBase.offset = 1000
        #expect(dflashCacheIsTrimmableAfterWriting(cache, plannedWriteCount: 2))
    }

    @Test
    func aRoundThatENDSOnTheRingBoundaryNeedsASnapshot() {
        // 1022 + 2 == 1024 == maxSize. `isTrimmable` is `offset < maxSize`, so
        // landing exactly on the boundary is already past it.
        let cache: [KVCache] = [RotatingKVCache(maxSize: 1024, keep: 0)]
        cache[0].asBase.offset = 1022
        #expect(canTrimPromptCache(cache))
        #expect(!dflashCacheIsTrimmableAfterWriting(cache, plannedWriteCount: 2))
    }

    @Test
    func aRoundThatCrossesTheRingBoundaryNeedsASnapshot() {
        let cache: [KVCache] = [RotatingKVCache(maxSize: 1024, keep: 0)]
        cache[0].asBase.offset = 1023
        // The exact shape of the defect: trimmable when the snapshot decision
        // is taken, untrimmable when the rollback needs it.
        #expect(canTrimPromptCache(cache))
        #expect(!dflashCacheIsTrimmableAfterWriting(cache, plannedWriteCount: 2))
    }

    @Test
    func anAlreadyWrappedCacheKeepsSnapshotting() {
        let cache: [KVCache] = [RotatingKVCache(maxSize: 1024, keep: 0)]
        cache[0].asBase.offset = 2048
        #expect(!canTrimPromptCache(cache))
        #expect(!dflashCacheIsTrimmableAfterWriting(cache, plannedWriteCount: 2))
    }

    @Test
    func cachesWithNoMaximumNeverNeedASnapshot() {
        // `KVCacheSimple` (the full-attention layers) declares no `maxSize`, so
        // it never wraps and a trim always works, however far decode has run.
        let cache: [KVCache] = [KVCacheSimple()]
        cache[0].asBase.offset = 50_000
        #expect(dflashCacheIsTrimmableAfterWriting(cache, plannedWriteCount: 16))
    }

    @Test
    func oneWrappingLayerDecidesForTheWholeList() {
        // Gemma 4 26B A4B: 25 sliding layers on a 1024 ring, 5 full-attention
        // layers with no ring. The rollback is all-or-nothing across the list,
        // so the sliding layers decide.
        let rotating = RotatingKVCache(maxSize: 1024, keep: 0)
        rotating.offset = 1023
        let simple = KVCacheSimple()
        simple.offset = 1023
        let cache: [KVCache] = [simple, rotating]
        #expect(!dflashCacheIsTrimmableAfterWriting(cache, plannedWriteCount: 2))
    }

    @Test
    func aWideCycleBlockWidensTheCrossingBand() {
        // A drafter round at depth 1 writes 2 positions, so only two offsets
        // per run can be the crossing round. A 16-wide cycle proposal writes
        // 16, so eight times as many offsets are. The cycle route does not
        // create the defect — the same round at width 2 is fine here — it only
        // makes the run more likely to land on it.
        let narrow = RotatingKVCache(maxSize: 1024, keep: 0)
        narrow.offset = 1010
        #expect(dflashCacheIsTrimmableAfterWriting([narrow], plannedWriteCount: 2))
        #expect(!dflashCacheIsTrimmableAfterWriting([narrow], plannedWriteCount: 16))
    }

    // MARK: - The failing receipt row, replayed

    @Test
    func theHumanEval94ShapeSnapshotsTheRoundThatUsedToThrow() throws {
        // HumanEval/94 in the receipt: 377 prompt tokens, dflash depth 1 (block
        // size 2), asked for 767 more tokens, died with `untrimmableCache`
        // after 6.15 s of a run whose peers finish in ~7.4 s — i.e. about 83%
        // of the way in, which is where committing 1024 - 377 = 647 tokens
        // falls. The rounds either side of the seam in the surviving rows all
        // commit 2, so the shape below is theirs with one rejecting round
        // placed exactly on the boundary.
        var script = [ScriptedRound]()
        // 646 tokens committed 2 at a time gets the frontier to 377 + 646 =
        // 1023, one position short of the ring.
        script.append(contentsOf: Array(repeating: .drafter(blockSize: 2, committed: 2), count: 323))
        // The round that crosses, and rejects its draft token.
        script.append(.drafter(blockSize: 2, committed: 1))
        script.append(contentsOf: Array(repeating: .drafter(blockSize: 2, committed: 2), count: 60))

        var run = ScriptedRun(promptTokenCount: 377, targetWindow: 1024, drafterWindow: 2047)
        let trace = try run.replay(script)

        let crossing = try #require(trace.firstIndex(where: { $0.snapshotted }))
        #expect(crossing == 323)
        #expect(trace[crossing].targetOffsetBeforeWrite == 1023)
        // The bug, stated as the test would have seen it: the old rule asked
        // only whether the cache was trimmable BEFORE the write, and at 1023 it
        // was — so this round got no snapshot, and its rejected token had
        // nowhere to go.
        #expect(trace[crossing].wouldHaveBeenTrimmableUnderTheOldRule)
        #expect(trace[crossing].rejected == 1)
        #expect(trace[crossing].rolledBackBySnapshot)
        // Every later round is already wrapped, so it snapshots too — that part
        // always worked, and 50 such rounds are visible in the surviving rows
        // of the same receipt.
        #expect(trace[(crossing + 1)...].allSatisfy { $0.snapshotted })
        // And the run completes: 647 + 120 committed tokens, cache offsets in
        // step with the committed chain the whole way.
        #expect(run.committedTotal == 767)
        #expect(run.targetOffset == 377 + 767)
    }

    // MARK: - Mixed sources across the seam

    @Test
    func mixedDrafterCycleAndTerminalRoundsKeepBothLedgersAligned() throws {
        // A run that exercises every round shape the session can produce, with
        // the seam in the middle of it: ordinary drafter rounds, a fully
        // accepted wide cycle round, a partially rejected wide cycle round, a
        // fully rejected drafter round, and a terminal round narrowed by the
        // token budget.
        var script = [ScriptedRound]()
        script.append(contentsOf: Array(repeating: .drafter(blockSize: 2, committed: 2), count: 300))
        script.append(.cycle(blockSize: 16, committed: 16))
        script.append(.cycle(blockSize: 16, committed: 16))
        script.append(.drafter(blockSize: 2, committed: 1))
        script.append(contentsOf: Array(repeating: .drafter(blockSize: 2, committed: 2), count: 8))
        script.append(.cycle(blockSize: 16, committed: 5))
        script.append(contentsOf: Array(repeating: .drafter(blockSize: 2, committed: 2), count: 20))
        script.append(.drafter(blockSize: 2, committed: 1))
        // Terminal round: the block is narrowed to what the token budget still
        // has room for, and every proposed token lands.
        script.append(.cycle(blockSize: 3, committed: 3))

        var run = ScriptedRun(promptTokenCount: 400, targetWindow: 1024, drafterWindow: 2047)
        let trace = try run.replay(script)

        // Exactly one round is the first to snapshot, and it is the one that
        // crosses the ring; nothing before it snapshots, nothing after it stops.
        let crossing = try #require(trace.firstIndex(where: { $0.snapshotted }))
        #expect(trace[..<crossing].allSatisfy { !$0.snapshotted })
        #expect(trace[crossing...].allSatisfy { $0.snapshotted })
        #expect(trace[crossing].targetOffsetBeforeWrite < 1024)
        #expect(
            trace[crossing].targetOffsetBeforeWrite + trace[crossing].blockSize >= 1024)

        // The drafter ledger, which is the OTHER thing a mixed run can get
        // wrong and the receipt was first suspected of: a drafter round leaves
        // the draft cache exactly on the committed frontier, and a cycle round
        // leaves it behind (never ahead), so the draft-cache trim in
        // `runDFlashGreedyRound` is a no-op on every round of this run.
        #expect(trace.allSatisfy { $0.extraDraftContext <= 0 })
        for entry in trace where entry.source == .drafter {
            #expect(entry.extraDraftContext == 0)
        }
        #expect(trace.contains { $0.source == .cycle && $0.extraDraftContext < 0 })
        // The debt a cycle round creates is repaid in full by the next drafter
        // round rather than accumulating across the run: what the drafter has
        // cached plus what it still owes is the committed chain, exactly, at
        // every point in a run of mixed sources.
        #expect(
            run.draftOffset + run.pendingDraftContextLength
                == run.promptTokenCount + run.committedTotal)
        #expect(trace.last(where: { $0.source == .drafter })?.extraDraftContext == 0)
    }

    @Test
    func aDrafterCacheWellPastItsOwnRingStillNeedsNoTrim() throws {
        // The drafter's rotating window is 2047 (`sliding_window` 2048 - 1), so
        // a long enough run wraps the DRAFT cache too. That is not a second
        // seam: the trim in `runDFlashGreedyRound` only fires when the draft
        // cache is AHEAD of the committed chain, and the session's context
        // ledger keeps it exactly on the chain no matter how far past 2047 the
        // run has gone.
        var script = Array(repeating: ScriptedRound.drafter(blockSize: 2, committed: 2), count: 1400)
        script.insert(.cycle(blockSize: 16, committed: 16), at: 700)

        var run = ScriptedRun(promptTokenCount: 400, targetWindow: 1024, drafterWindow: 2047)
        let trace = try run.replay(script)

        #expect(run.draftOffset > 2047)
        #expect(!canTrimPromptCache(run.draftCache))
        #expect(trace.allSatisfy { $0.extraDraftContext <= 0 })
    }
}

// MARK: - Scripted run

private enum ScriptedSource: Equatable {
    case drafter
    case cycle
}

/// One scripted round: where the block came from, how wide it is, and how many
/// tokens the target committed out of it (`accepted + 1`).
private struct ScriptedRound {
    let source: ScriptedSource
    let blockSize: Int
    let committed: Int

    static func drafter(blockSize: Int, committed: Int) -> ScriptedRound {
        ScriptedRound(source: .drafter, blockSize: blockSize, committed: committed)
    }

    static func cycle(blockSize: Int, committed: Int) -> ScriptedRound {
        ScriptedRound(source: .cycle, blockSize: blockSize, committed: committed)
    }
}

private struct ScriptedRoundTrace {
    let source: ScriptedSource
    let blockSize: Int
    let rejected: Int
    let targetOffsetBeforeWrite: Int
    let snapshotted: Bool
    let wouldHaveBeenTrimmableUnderTheOldRule: Bool
    let rolledBackBySnapshot: Bool
    /// `draftCache.offset - (promptTokenCount + generatedTokenCount - 1)` at the
    /// point `runDFlashGreedyRound` reads it — positive means the round would
    /// have to trim the draft cache.
    let extraDraftContext: Int
}

private enum ScriptedRunError: Error {
    case cacheDivergedFromTheCommittedChain(round: Int, offset: Int, expected: Int)
    case rollbackImpossible(round: Int)
}

/// Replays the two ledgers a single-stream DFlash free run keeps, against the
/// real cache types and the real trim helpers.
///
/// The two advance rules are the engine's, and are facts about the MLX forwards
/// rather than about this file:
///   * the target verify writes `blockSize` positions and the rollback removes
///     the rejected tail (`DFlashGreedyRound`),
///   * `draftBlock` advances the draft cache by the length of the context it is
///     handed, and never by the block it proposes (`DFlashDraftModel`).
private struct ScriptedRun {
    let promptTokenCount: Int
    private(set) var targetCache: [KVCache]
    private(set) var draftCache: [KVCache]
    private(set) var committedTotal = 0
    /// Committed hidden the drafter has not cached yet, in tokens. Mirrors
    /// `RuntimeWorkerDFlashFreeRunSession.pendingDraftContext`, whose entries
    /// are `[1, committed, taps * H]` slices.
    private(set) var pendingDraftContextLength: Int

    init(promptTokenCount: Int, targetWindow: Int, drafterWindow: Int) {
        self.promptTokenCount = promptTokenCount
        // One sliding layer and one full-attention layer, the two kinds Gemma 4
        // and the z-lab drafter both build.
        let targetRotating = RotatingKVCache(maxSize: targetWindow, keep: 0)
        let targetSimple = KVCacheSimple()
        targetRotating.offset = promptTokenCount
        targetSimple.offset = promptTokenCount
        self.targetCache = [targetRotating, targetSimple]

        // The draft cache holds no prompt: `draftBlock` caches the projected
        // target hidden, and the first round is what hands it the prompt's.
        let draftRotating = RotatingKVCache(maxSize: drafterWindow, keep: 0)
        let draftSimple = KVCacheSimple()
        self.draftCache = [draftRotating, draftSimple]
        self.pendingDraftContextLength = promptTokenCount
    }

    var targetOffset: Int { targetCache[0].offset }
    var draftOffset: Int { draftCache[0].offset }

    mutating func replay(_ script: [ScriptedRound]) throws -> [ScriptedRoundTrace] {
        var trace = [ScriptedRoundTrace]()
        trace.reserveCapacity(script.count)
        for (index, round) in script.enumerated() {
            trace.append(try step(round, index: index))
        }
        return trace
    }

    private mutating func step(_ round: ScriptedRound, index: Int) throws -> ScriptedRoundTrace {
        // 1. The snapshot decision, taken before anything is written.
        let offsetBefore = targetOffset
        let preWriteOffsets = targetCache.map { $0.offset }
        let oldRuleSaidTrimmable = canTrimPromptCache(targetCache)
        let snapshotted = !dflashCacheIsTrimmableAfterWriting(
            targetCache, plannedWriteCount: round.blockSize)
        let snapshot = snapshotted ? targetCache.map { $0.copy() } : nil

        // 2. The drafter half of the round. A drafter round consumes every
        //    pending context vector; a cycle round calls no drafter at all.
        //    The frontier `runDFlashGreedyRound` compares against is
        //    `promptTokenCount + generatedTokenCount - 1`, and
        //    `generatedTokenCount` counts the seed plus everything committed so
        //    far.
        if round.source == .drafter {
            for cache in draftCache {
                cache.asBase.offset += pendingDraftContextLength
            }
            pendingDraftContextLength = 0
        }
        let generatedTokenCount = 1 + committedTotal
        let committedFrontier = promptTokenCount + generatedTokenCount - 1
        let extraDraftContext = draftOffset - committedFrontier

        // 3. The verify writes the whole block.
        for cache in targetCache {
            cache.asBase.offset += round.blockSize
        }

        // 4. The rollback.
        let rejected = round.blockSize - round.committed
        var rolledBackBySnapshot = false
        if rejected > 0 {
            var trimmed = 0
            if canTrimPromptCache(targetCache) {
                trimmed = trimPromptCache(targetCache, numTokens: rejected)
            }
            if trimmed != rejected {
                guard let snapshot else {
                    throw ScriptedRunError.rollbackImpossible(round: index)
                }
                // The engine restores the snapshot and replays the accepted
                // prefix through the target, which lands the caches at
                // `pre-write + committed`. The offsets are re-seeded here
                // rather than read back off the copies because a `KVCacheSimple`
                // holding no arrays cannot round-trip its offset through
                // `state` — an artifact of running this without MLX, not of the
                // runtime, where the state getter slices to `offset` and the
                // setter restores it.
                targetCache = snapshot
                for (cache, offset) in zip(targetCache, preWriteOffsets) {
                    cache.asBase.offset = offset + round.committed
                }
                rolledBackBySnapshot = true
            }
        }

        // 5. Commit, and repay the drafter-context ledger.
        committedTotal += round.committed
        pendingDraftContextLength += round.committed

        let expected = promptTokenCount + committedTotal
        for cache in targetCache where cache.offset != expected {
            throw ScriptedRunError.cacheDivergedFromTheCommittedChain(
                round: index, offset: cache.offset, expected: expected)
        }

        return ScriptedRoundTrace(
            source: round.source,
            blockSize: round.blockSize,
            rejected: rejected,
            targetOffsetBeforeWrite: offsetBefore,
            snapshotted: snapshotted,
            wouldHaveBeenTrimmableUnderTheOldRule: oldRuleSaidTrimmable,
            rolledBackBySnapshot: rolledBackBySnapshot,
            extraDraftContext: extraDraftContext)
    }
}

extension KVCache {
    /// `offset` is settable on `BaseKVCache`, which every cache in this tree
    /// derives from; the protocol exposes it read-only. Seeding an offset is
    /// how these tests place a cache at a ring position without allocating the
    /// arrays a real prefill would.
    fileprivate var asBase: BaseKVCache {
        guard let base = self as? BaseKVCache else {
            preconditionFailure("DFlash caches derive from BaseKVCache")
        }
        return base
    }
}
