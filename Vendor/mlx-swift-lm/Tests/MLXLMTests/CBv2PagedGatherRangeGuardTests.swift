// CBv2PagedGatherRangeGuardTests.swift
//
// What `PagedSequenceKV.gatherRange`'s eviction precondition actually
// protects — pinned from both sides, at gemma-4's shipping geometry.
//
// THIS FILE EXISTS BECAUSE THE GUARD LOOKS WRONG AND IS NOT. It is expressed
// in TOKENS (`start >= oldest resident position`) while the loop underneath
// it walks PAGES, so it reads like a latent aliasing bug: at pageSize 16 and
// a 65-page ring, a 1,026-token gather starting at `15 (mod 16)` spans 66
// logical pages and `lp % 65` maps its first and last page to the SAME
// physical page. `ringWrapIsNotAnAliasEvenAtSixtySixPages` is the disproof.
// The ring is a flat circular buffer of `ringPages * pageSize` TOKEN slots
// that happens to be paged — `p` lives at `(table[(p / pageSize) %
// ringPages], p % pageSize)`, a bijection onto `p % (ringPages * pageSize)`
// — so the page list always names the CANONICAL slot of every requested
// position, and the duplicated page serves the two ends of the range from
// disjoint slots. Tightening the guard to `lpLast - lpFirst + 1 <=
// ringPages` would abort the daemon on byte-exact reads.
//
// The guard WAS one speculative span short of exact, in a direction the page
// arithmetic cannot see: it measured from `absoluteOffset`, and a rolled-back
// round leaves the rejected draft's K/V in the ring while the cursor
// retreats. `rejectedDraftKVIsNotResidentJustBecauseTheCursorRetreated` is
// that hazard, reproduced. The fix is `oldestValidPosition`, measured from
// `writtenHighWater` — which is exactly what the contiguous backend's
// `CBv2WindowedSequenceKV.oldestValidPosition` has always done.
//
// TRAPS ARE NOT TESTABLE. A failed `precondition` is a process abort, so no
// test here may call `gatherRange` past its guard. Each hazard is therefore
// pinned as a PAIR: `unguardedGather` rebuilds the exact read the guard is
// blocking and shows what it returns, and a separate assertion shows the
// guard refuses exactly that start. `gatherMirrorsGatherRangeOnLegalRanges`
// keeps the rebuild honest.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedGatherRangeGuard")
struct CBv2PagedGatherRangeGuardTests {

    // MARK: - gemma-4 geometry

    /// window 1,024 / chunk 512 / pageSize 16 / span 8 => 65 pages, 1,040
    /// tokens. Derived through `PagedKVPool.ringPageCount`, never hard-coded,
    /// so a sizing change fails the sizing tests rather than these.
    private static let window = 1024
    private static let chunk = 512
    private static let kvHeads = 2
    private static let headDim = 64

    /// fp16 represents integers exactly only up to 2,048. Every canary here
    /// is a position, so every fixture must stay under that or a rounded
    /// canary reports a mismatch that is the test's fault, not the code's.
    private static let canaryLimit = 2048

    private func config() -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: 32 << 20, maxPrefillChunk: Self.chunk,
            nominalMaxSequenceLength: 8192)
    }

    private var pageSize: Int { config().pageSize }
    private var ringPages: Int { PagedKVPool.ringPageCount(window: Self.window, config: config()) }
    private var ringTokens: Int { ringPages * pageSize }

    private func windowedKinds() -> [CBv2LayerKind] {
        [
            CBv2LayerKind(
                attention: .slidingWindow(Self.window), headDim: Self.headDim,
                kvHeads: Self.kvHeads, queryHeads: Self.kvHeads * 2)
        ]
    }

    private func fullKinds() -> [CBv2LayerKind] {
        [
            CBv2LayerKind(
                attention: .full, headDim: Self.headDim, kvHeads: Self.kvHeads,
                queryHeads: Self.kvHeads * 2)
        ]
    }

    // MARK: - Fixture

    private struct Fixture {
        let backend: PagedKVBackend
        let state: [CBv2SequenceKV?]
        let row: PagedSequenceKV
        func release() { backend.release(state) }
    }

    private func makeRow(windowed: Bool = true, maxLength: Int = 4096) throws -> Fixture {
        let kinds = windowed ? windowedKinds() : fullKinds()
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: maxLength)
        return Fixture(
            backend: backend, state: state, row: try #require(state[0] as? PagedSequenceKV))
    }

    // MARK: - Canaries

    /// Every element of token column `p` carries the value `p`, so a column
    /// served from the wrong position NAMES the position it came from
    /// instead of merely comparing unequal.
    private func canary(_ values: [Float]) -> (keys: MLXArray, values: MLXArray) {
        var flat = [Float]()
        flat.reserveCapacity(Self.kvHeads * values.count * Self.headDim)
        for _ in 0 ..< Self.kvHeads {
            for value in values {
                for _ in 0 ..< Self.headDim { flat.append(value) }
            }
        }
        let k = MLXArray(flat, [Self.kvHeads, values.count, Self.headDim]).asType(.float16)
        return (k, -k)
    }

    private func canary(_ positions: Range<Int>) -> (keys: MLXArray, values: MLXArray) {
        canary(positions.map { Float($0) })
    }

    /// Fill `[absoluteOffset, end)` with position canaries via `write`,
    /// respecting `maxPrefillChunk`.
    private func writeCanaries(_ row: PagedSequenceKV, through end: Int) {
        #expect(end <= Self.canaryLimit, "fixture exceeds fp16's exact integer range")
        while row.absoluteOffset < end {
            let n = min(Self.chunk, end - row.absoluteOffset)
            let (k, v) = canary(row.absoluteOffset ..< row.absoluteOffset + n)
            row.write(keys: k, values: v)
        }
    }

    /// Per-column canary values of a `[1, kvHeads, count, headDim]` gather.
    private func columns(_ gathered: MLXArray) -> [Float] {
        gathered[0, 0, 0..., 0].asType(.float32).asArray(Float.self)
    }

    /// First column whose canary is not its own position, or nil.
    private func firstSubstitution(
        _ gathered: MLXArray, _ expected: Range<Int>
    ) -> (column: Int, got: Float, wanted: Int)? {
        let got = columns(gathered)
        guard got.count == expected.count else {
            return (column: -1, got: Float(got.count), wanted: expected.count)
        }
        for (i, p) in expected.enumerated() where got[i] != Float(p) {
            return (column: i, got: got[i], wanted: p)
        }
        return nil
    }

    private func expectExact(
        _ gathered: MLXArray, _ expected: Range<Int>, _ label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        if let bad = firstSubstitution(gathered, expected) {
            Issue.record(
                "\(label): column \(bad.column) holds position \(bad.got), wanted \(bad.wanted)",
                sourceLocation: sourceLocation)
        }
    }

    /// The gather `gatherRange` would perform, with the guard bypassed.
    ///
    /// This MIRRORS `gatherRange`'s page-list construction deliberately: a
    /// failed `precondition` is a process abort, so observing what the guard
    /// blocks is only possible by rebuilding the read beside it.
    /// `gatherMirrorsGatherRangeOnLegalRanges` asserts the mirror is faithful.
    private func unguardedGather(_ row: PagedSequenceKV, start: Int, count: Int) -> MLXArray {
        let s = row.pool.config.pageSize
        let lpFirst = start / s
        let lpLast = (start + count - 1) / s
        var pages: [Int32] = []
        pages.reserveCapacity(lpLast - lpFirst + 1)
        for lp in lpFirst ... lpLast {
            pages.append(row.table[row.ringPages.map { lp % $0 } ?? lp])
        }
        return row.pool.gather(
            group: row.groupKey, pages: pages, firstSlot: start % s, count: count
        ).keys
    }

    // MARK: - The page-span hazard that is not one

    /// 1,026 tokens from `start == 15`: 66 logical pages through a 65-page
    /// ring, first and last page on the SAME physical page. Every column is
    /// still its own position.
    ///
    /// The duplicated page is the buffer wrapping. Its first occurrence
    /// serves slots `[15, 16)` (the old positions, not yet overwritten) and
    /// its last serves slot `[0, 0]` (the new one, just written) — disjoint,
    /// and both canonical.
    @Test func ringWrapIsNotAnAliasEvenAtSixtySixPages() throws {
        let fixture = try makeRow()
        defer { fixture.release() }
        let row = fixture.row
        let s = pageSize
        #expect(ringPages == 65, "gemma-4 ring")
        #expect(ringTokens == 1040)

        let start = s - 1
        let count = 1026
        writeCanaries(row, through: start + count)
        #expect(row.absoluteOffset == 1041)

        let lpFirst = start / s
        let lpLast = (start + count - 1) / s
        #expect(lpLast - lpFirst + 1 == ringPages + 1, "the range spans one page more than the ring")
        #expect(lpFirst % ringPages == lpLast % ringPages, "first and last page share a ring slot")
        #expect(
            row.table[lpFirst % ringPages] == row.table[lpLast % ringPages],
            "and therefore the same physical page")
        // Disjointness, which is what makes the duplicate harmless.
        #expect((start + count - 1) % s < start % s, "wrapped slot ranges overlap")

        // Legal under the guard, and byte-exact.
        #expect(start >= row.oldestValidPosition)
        let (k, v) = row.gatherRange(start: start, count: count)
        expectExact(k, start ..< start + count, "66-page wrap")
        let expected = canary(start ..< start + count)
        #expect(
            arrayEqual(k, expected.keys.expandedDimensions(axis: 0)).item(Bool.self),
            "keys are not byte-exact")
        #expect(
            arrayEqual(v, expected.values.expandedDimensions(axis: 0)).item(Bool.self),
            "values are not byte-exact")
    }

    /// The whole ring is gatherable at EVERY alignment — the margin over the
    /// widest shipping ask is 16 tokens, not zero pages.
    ///
    /// A page-span guard would reject 15 of these 16 alignments (every one
    /// whose start is not page-aligned spans `ringPages + 1`), all of which
    /// return exact data here.
    @Test func theWholeRingIsGatherableAtEveryAlignment() throws {
        let s = pageSize
        for r in 0 ..< s {
            let fixture = try makeRow()
            defer { fixture.release() }
            let row = fixture.row
            writeCanaries(row, through: 1536 + r)
            let start = row.absoluteOffset - ringTokens
            #expect(start % s == r, "fixture did not produce alignment \(r)")
            #expect(start == row.oldestValidPosition, "the whole ring must be legal")

            let spanPages = (start + ringTokens - 1) / s - start / s + 1
            #expect(spanPages == (r == 0 ? ringPages : ringPages + 1))
            let (k, _) = row.gatherRange(start: start, count: ringTokens)
            expectExact(k, start ..< start + ringTokens, "alignment \(r)")
        }
    }

    /// `unguardedGather` must be the same read `gatherRange` performs, or
    /// the hazard tests below prove nothing.
    @Test func gatherMirrorsGatherRangeOnLegalRanges() throws {
        let fixture = try makeRow()
        defer { fixture.release() }
        let row = fixture.row
        writeCanaries(row, through: 1536)
        for (start, count) in [
            (row.absoluteOffset - ringTokens, ringTokens),
            (513, 1000),
            (row.absoluteOffset - 1, 1),
        ] {
            let (guarded, _) = row.gatherRange(start: start, count: count)
            let mirrored = unguardedGather(row, start: start, count: count)
            #expect(
                arrayEqual(guarded, mirrored).item(Bool.self),
                "mirror diverged at start \(start), count \(count)")
        }
    }

    // MARK: - The hazard that is real

    /// A rolled-back speculative round leaves the REJECTED draft's K/V in
    /// the ring, and the cursor retreating does not bring back what those
    /// writes destroyed.
    ///
    /// This is what the guard was actually short of. Measured from
    /// `absoluteOffset` — the expression this guard used to carry — the
    /// oldest `maxSpeculativeSpan` tokens of the ring read back as the
    /// rejected draft, silently. Measured from `writtenHighWater` they are
    /// outside the guard.
    @Test func rejectedDraftKVIsNotResidentJustBecauseTheCursorRetreated() throws {
        let fixture = try makeRow()
        defer { fixture.release() }
        let row = fixture.row
        let span = CBv2PagedSpeculation.maxSpeculativeSpan

        writeCanaries(row, through: 1536)
        let frontier = row.absoluteOffset
        row.beginSpeculativeWrite()
        // Rejected draft: canaries no real position can produce.
        let draft = canary((0 ..< span).map { Float(-1 - $0) })
        row.write(keys: draft.keys, values: draft.values)
        row.rollback(span)
        row.commitSpeculativeWrite()

        #expect(row.absoluteOffset == frontier, "rollback restored the cursor")
        #expect(row.writtenHighWater == frontier + span, "the writes still happened")

        // What the old expression admitted.
        let relaxed = row.absoluteOffset - ringTokens
        #expect(relaxed < row.oldestValidPosition, "the guard did not tighten")
        #expect(
            row.oldestValidPosition - relaxed == span,
            "the band the old guard admitted is exactly one speculative span")

        // And what it would have handed back: the rejected draft, at
        // positions `ringTokens` older than the ones it was drafted for.
        let poisoned = unguardedGather(row, start: relaxed, count: ringTokens)
        let got = columns(poisoned)
        for i in 0 ..< span {
            #expect(
                got[i] == Float(-1 - i),
                "column \(i) should hold rejected draft token \(i), holds \(got[i])")
        }
        #expect(
            firstSubstitution(poisoned, relaxed ..< relaxed + ringTokens) != nil,
            "the relaxed range must be observably wrong, or this test proves nothing")

        // The tightened range is exact, and legal.
        let start = row.oldestValidPosition
        let (k, _) = row.gatherRange(start: start, count: row.absoluteOffset - start)
        expectExact(k, start ..< row.absoluteOffset, "post-rollback residency")
    }

    /// The guard's threshold is EXACTLY where the data goes wrong — one
    /// token looser is corruption, and it is not one token tighter than it
    /// needs to be.
    ///
    /// Scans for the oldest start whose read is exact and pins that to
    /// `oldestValidPosition`, quiescent and after a rolled-back round. This
    /// is the assertion that keeps the guard from drifting in EITHER
    /// direction: a looser one returns substituted KV, a tighter one aborts
    /// the daemon on reads that are correct.
    @Test(arguments: [0, CBv2PagedSpeculation.maxSpeculativeSpan])
    func oldestValidPositionIsExactlyWhereTheDataGoesWrong(rolledBack: Int) throws {
        let fixture = try makeRow()
        defer { fixture.release() }
        let row = fixture.row

        writeCanaries(row, through: 1536)
        let frontier = row.absoluteOffset
        if rolledBack > 0 {
            row.beginSpeculativeWrite()
            let draft = canary((0 ..< rolledBack).map { Float(-1 - $0) })
            row.write(keys: draft.keys, values: draft.values)
            row.rollback(rolledBack)
            row.commitSpeculativeWrite()
        }

        var firstExact: Int?
        let floor = frontier - ringTokens
        for start in floor ... (floor + rolledBack + 2) {
            let read = unguardedGather(row, start: start, count: row.absoluteOffset - start)
            if firstSubstitution(read, start ..< row.absoluteOffset) == nil {
                firstExact = start
                break
            }
        }
        let where_ = firstExact.map(String.init) ?? "never"
        #expect(
            firstExact == row.oldestValidPosition,
            "data becomes exact at \(where_), guard says \(row.oldestValidPosition) (rolled back \(rolledBack))")
    }

    // MARK: - Caller sweep

    /// Every `gatherRange` caller, at its widest, against the guard.
    ///
    /// There are four, all inside this module (`gatherRange` is internal):
    ///
    ///  1. `PagedSequenceKV.attendableViews()` — `retainedCount`, i.e.
    ///     `maxWindowExposure(window) == window == 1,024`. THE WIDEST.
    ///  2. `PagedSequenceKV.update` — the pre-write history,
    ///     `absoluteOffset - max(baseOffset, absoluteOffset - (window - 1))`,
    ///     so at most `window - 1 == 1,023`.
    ///  3. `PagedLayerCache.prefillKVWritingChunk`, ordinary branch —
    ///     `queryStart - max(baseOffset, queryStart - window + 1)`, the same
    ///     `window - 1 == 1,023`, and the same 17 tokens of margin.
    ///  4. `PagedLayerCache.prefillKVWritingChunk`, frozen branch —
    ///     `[baseOffset, queryStart + chunk)`, unbounded by any window, but
    ///     `adoptFrozen` refuses windowed rows so the row has no ring at all
    ///     (covered by `fullRowsEvictNothingSoTheGuardIsInert`).
    ///
    /// (1) and (2) are driven live below; (3) is (2)'s arithmetic on the
    /// layer path and is exercised by `CBv2PagedQueryBlockingTests` /
    /// `CBv2PagedSeamContractTests`.
    @Test func theWidestShippingCallersClearTheGuard() throws {
        let fixture = try makeRow()
        defer { fixture.release() }
        let row = fixture.row

        // Caller 2, at its widest: `update` in `maxPrefillChunk` chunks.
        var written = 0
        while written < 1536 {
            let n = min(Self.chunk, 1536 - written)
            let before = row.absoluteOffset
            let historyStart = max(row.baseOffset, before - (Self.window - 1))
            #expect(
                historyStart >= row.oldestValidPosition,
                "update's history gather is evicted at frontier \(before)")
            let (k, _) = row.update(
                keys: canary(before ..< before + n).keys,
                values: canary(before ..< before + n).values)
            written += n
            #expect(k.dim(2) == (before - historyStart) + n, "update's returned width")
            expectExact(k, historyStart ..< row.absoluteOffset, "update view at \(before)")
            #expect(before - historyStart <= Self.window - 1, "update out-ran window - 1")
        }

        // Caller 1, at its widest: the full retained window.
        #expect(row.retainedCount == Self.window)
        let attendStart = row.absoluteOffset - row.retainedCount
        #expect(attendStart >= row.oldestValidPosition)
        let (attendK, _) = row.attendableViews()
        expectExact(attendK, attendStart ..< row.absoluteOffset, "attendableViews")

        // The margin, in tokens, for the widest caller there is.
        #expect(
            attendStart - row.oldestValidPosition == ringTokens - Self.window,
            "quiescent margin")
        #expect(ringTokens - Self.window == 16, "gemma-4 quiescent margin is 16 tokens")
    }

    /// The margin the widest caller actually has, and what eats it.
    ///
    /// 16 tokens quiescent; `maxSpeculativeSpan` of that is spent by a
    /// rolled-back round, leaving 8. Re-inflating `retainedCount` past
    /// `ringTokens - maxSpeculativeSpan` (1,032) — for instance back to the
    /// phase-dependent `window - 1 + lastUpdateTokens` this row used to
    /// report — turns `attendableViews()` into a gather of positions the ring
    /// no longer holds. This test goes red at that point; without it the
    /// symptom is substituted KV on a hot path with no trap.
    @Test func theWidestCallerKeepsEightTokensAfterARolledBackRound() throws {
        let fixture = try makeRow()
        defer { fixture.release() }
        let row = fixture.row
        let span = CBv2PagedSpeculation.maxSpeculativeSpan

        writeCanaries(row, through: 1536)
        #expect(
            row.absoluteOffset - row.retainedCount - row.oldestValidPosition
                == ringTokens - Self.window,
            "quiescent margin for attendableViews()")

        row.beginSpeculativeWrite()
        let draft = canary((0 ..< span).map { Float(-1 - $0) })
        row.write(keys: draft.keys, values: draft.values)
        row.rollback(span)
        row.commitSpeculativeWrite()

        let margin = row.absoluteOffset - row.retainedCount - row.oldestValidPosition
        #expect(margin == ringTokens - Self.window - span, "post-round margin")
        #expect(margin == 8, "gemma-4 post-round margin is 8 tokens")
        #expect(margin >= 0, "attendableViews() now gathers evicted positions")

        // And it still reads exactly, which is the point of the margin.
        let start = row.absoluteOffset - row.retainedCount
        let (k, _) = row.attendableViews()
        expectExact(k, start ..< row.absoluteOffset, "attendableViews after a rolled-back round")
    }

    /// Full rows have no ring: nothing is ever overwritten, so the guard
    /// floors at `baseOffset` and never fires. This is the frozen prefill
    /// branch's whole safety argument (`adoptFrozen` refuses windowed rows).
    @Test func fullRowsEvictNothingSoTheGuardIsInert() throws {
        let fixture = try makeRow(windowed: false, maxLength: 2048)
        defer { fixture.release() }
        let row = fixture.row
        #expect(row.ringPages == nil)

        writeCanaries(row, through: 1536)
        #expect(row.oldestValidPosition == row.baseOffset)
        #expect(row.retainedCount == 1536, "a full row retains everything")
        let (k, _) = row.gatherRange(start: 0, count: row.absoluteOffset)
        expectExact(k, 0 ..< row.absoluteOffset, "full-row gather")

        // A rollback frees the tail pages; the surviving prefix stays
        // gatherable from position 0, high water notwithstanding.
        row.rollback(64)
        #expect(row.writtenHighWater == 1536)
        #expect(row.oldestValidPosition == 0)
        let (k2, _) = row.gatherRange(start: 0, count: row.absoluteOffset)
        expectExact(k2, 0 ..< row.absoluteOffset, "full-row gather after rollback")
    }

    // MARK: - The precondition itself

    /// The guard is a `precondition`, so every in-process test above can
    /// only assert its THRESHOLD — delete the `precondition` line and they
    /// all still pass. These two fail, because they run the gather in a
    /// spawned process and require that process to die, or not to.
    ///
    /// THE PAIR IS THE POINT. `.failure` matches ANY non-zero exit, so a
    /// lone death test proves nothing: a child that dies of a Metal failure,
    /// a fixture precondition, or an unreleased row's `deinit` assert passes
    /// it just as well as the guard does. The two bodies here differ by
    /// exactly the `maxSpeculativeSpan` tokens that separate evicted from
    /// resident, so anything environmental takes the control down with it.
    ///
    /// Verified by mutation, not by argument: with `oldestValidPosition`
    /// reverted to the old `absoluteOffset`-relative expression, the
    /// `.failure` test goes red (the child completes the gather and exits 0)
    /// while the control stays green.
    @Test func gatheringIntoARolledBackRoundTraps() async throws {
        await #expect(processExitsWith: .failure) {
            try CBv2PagedGatherRangeGuardTests.gatherAfterARolledBackRound(aboveTheRingFloor: 0)
        }
    }

    /// Control for the test above: the same fixture and the same process
    /// shape, reading from one speculative span later — the oldest position
    /// the row still holds. Must exit CLEANLY.
    ///
    /// The span is spelled out inline rather than bound to a `let`: an exit
    /// test body is lowered to a C function pointer, so capturing even one
    /// local is a compile error.
    @Test func gatheringAtOldestValidPositionDoesNotTrap() async throws {
        await #expect(processExitsWith: .success) {
            try CBv2PagedGatherRangeGuardTests.gatherAfterARolledBackRound(
                aboveTheRingFloor: CBv2PagedSpeculation.maxSpeculativeSpan)
        }
    }

    /// Shared body. STATIC and self-contained: an exit test body runs in a
    /// fresh process and may capture nothing, not even `self`. Contents are
    /// irrelevant — only the exit status is — so this writes zeros and skips
    /// the canaries.
    ///
    /// `aboveTheRingFloor` is measured from `absoluteOffset - ringTokens`,
    /// which is where the guard used to sit and where the ring floors when
    /// nothing was ever rolled back. After the round below, the row's true
    /// floor is one span above it.
    ///
    /// THE RELEASE AT THE END IS LOAD-BEARING, not tidiness:
    /// `PagedSequenceKV.deinit` asserts on an unreleased row, so without it
    /// the child aborts on the clean path too. It also makes the result
    /// build-configuration INDEPENDENT — that check is an `assert`, compiled
    /// out at `-O`, so an unreleased row would abort a debug child and exit
    /// 0 in a release one.
    private static func gatherAfterARolledBackRound(aboveTheRingFloor delta: Int) throws {
        let cfg = PagedKVPoolConfig(
            capacityBytes: 32 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 8192)
        let kinds = [
            CBv2LayerKind(
                attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
                queryHeads: kvHeads * 2)
        ]
        let backend = try PagedKVBackend(layerKinds: kinds, config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 4096)
        guard let row = state[0] as? PagedSequenceKV else {
            fatalError("paged backend did not produce a paged row")
        }
        let ringTokens = PagedKVPool.ringPageCount(window: window, config: cfg) * cfg.pageSize
        while row.absoluteOffset < 1536 {
            let n = min(chunk, 1536 - row.absoluteOffset)
            let block = MLXArray.zeros([kvHeads, n, headDim], dtype: .float16)
            row.write(keys: block, values: block)
        }
        let span = CBv2PagedSpeculation.maxSpeculativeSpan
        row.beginSpeculativeWrite()
        let draft = MLXArray.zeros([kvHeads, span, headDim], dtype: .float16)
        row.write(keys: draft, values: draft)
        row.rollback(span)
        row.commitSpeculativeWrite()
        let start = row.absoluteOffset - ringTokens + delta
        _ = row.gatherRange(start: start, count: row.absoluteOffset - start)
        backend.release(state)
    }
}
