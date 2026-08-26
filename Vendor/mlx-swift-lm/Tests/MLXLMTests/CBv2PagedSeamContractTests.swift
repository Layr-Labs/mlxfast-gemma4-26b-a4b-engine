// CBv2PagedSeamContractTests.swift
//
// The mechanical half of `PagedSeamContract.swift`. That file is the frozen
// seam three concurrent tracks build against; this file is what stops it
// drifting away from the code it describes.
//
// Two of these suites exist because the frozen contract HAD drifted (PR#86
// review):
//
//  * the contract declared a ring formula that `PagedKVPool` never
//    implemented, and nothing failed when they disagreed. `RingFormula`
//    below fails.
//  * the contract froze `restoreWindow(keys:values:base:)`, which cannot
//    tell whether the window it is handed belongs at the boundary being
//    adopted. `WindowSnapshotBoundary` below pins the refusal.
//
// The third is the always-run copy of an invariant that used to be guarded
// by an `assert` inside a function nothing called.
//
// NOTE ON THE RING SUITE. The formula the contract used to declare —
// `ceil(window / pageSize) + span pages`, 65 for gemma-4 — WAS wrong when it
// was written, and shipping it aborted the daemon in ordinary windowed
// prefill. It is right now, and the tests below are deliberately written so
// that they fail again if the reason it became right is undone. They pin the
// CONDITION, not the number: `attendableTokens` may drop the prefill-chunk
// term only while both write paths gather before they write.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// MARK: - Ring formula

@Suite("CBv2 paged seam contract: ring formula")
struct CBv2PagedSeamContractRingFormulaTests {

    /// Windows, page sizes and chunks wide enough to cross every rounding
    /// boundary in either expression: exact multiples, one-off-either-side,
    /// a page larger than the window, and a chunk larger than the window.
    private static let windows = [1, 15, 16, 17, 32, 128, 512, 1024, 4096]
    private static let pageSizes = [1, 8, 16, 64, 256]
    private static let chunks = [1, 8, 15, 16, 17, 64, 512, 4096]

    private func config(pageSize: Int, maxPrefillChunk: Int) -> PagedKVPoolConfig {
        // `ringPageCount` reads only `pageSize` and `maxPrefillChunk`; the
        // rest is inert here, and `maxBufferLength` is passed explicitly so
        // the matrix never touches the GPU.
        PagedKVPoolConfig(
            pageSize: pageSize,
            capacityBytes: 1 << 20,
            maxPrefillChunk: maxPrefillChunk,
            nominalMaxSequenceLength: 4096,
            maxBufferLength: 1 << 40)
    }

    /// THE binding test. `CBv2PagedRingGeometry.ringPageCount` is the
    /// contract's statement of the ring formula and
    /// `PagedKVPool.ringPageCount` is the shipping one; if either moves
    /// without the other, this fails.
    ///
    /// ONE TERM IS SHARED, SO THIS TEST IS BLIND TO IT. Both sides reach
    /// `PagedSequenceKV.maxWindowExposure(window:)` for the exposure term —
    /// the contract through `CBv2PagedRingGeometry.attendableTokens`, the
    /// pool directly — so a change there moves both derivations together and
    /// this comparison still passes. That is deliberate: the coupling is the
    /// safety property (widen what a row exposes and the ring grows with it,
    /// rather than the row out-running a ring sized from a stale literal),
    /// so breaking it here to make the derivations independent would trade a
    /// real invariant for test coverage.
    ///
    /// The exposure term is instead pinned separately, immediately below.
    /// It is the term the whole 65-page argument rests on, so it gets its
    /// own assertion rather than riding on this one.
    @Test func ringFormulaMatchesPagedKVPool() {
        for window in Self.windows {
            for pageSize in Self.pageSizes {
                for chunk in Self.chunks {
                    let cfg = config(pageSize: pageSize, maxPrefillChunk: chunk)
                    let contract = CBv2PagedRingGeometry.ringPageCount(
                        window: window, pageSize: pageSize, maxPrefillChunk: chunk)
                    let shipping = PagedKVPool.ringPageCount(window: window, config: cfg)
                    #expect(
                        contract == shipping,
                        """
                        window \(window), pageSize \(pageSize), chunk \(chunk): the frozen \
                        contract says \(contract) pages, PagedKVPool.ringPageCount computes \
                        \(shipping). PagedSeamContract.swift and PagedKVPool.swift must state \
                        one formula — update both, or neither.
                        """)
                }
            }
        }
    }

    /// The exposure term, pinned on its own because
    /// `ringFormulaMatchesPagedKVPool` cannot see it.
    ///
    /// `maxWindowExposure(window:) == window` is the load-bearing claim of
    /// the whole 65-page argument: it is `window` rather than
    /// `window - 1 + maxPrefillChunk` ONLY because both write paths gather
    /// BEFORE they write. Re-introduce a post-write gather anywhere and this
    /// number has to grow — at which point gemma-4's windowed layers go back
    /// from 65 pages to something near the pre-shrink 97, on 25 of 30
    /// layers. That is a memory decision, so it must fail a test and be
    /// looked at, not ride silently through both derivations at once.
    @Test func windowExposureIsExactlyTheWindow() {
        for window in Self.windows {
            #expect(
                PagedSequenceKV.maxWindowExposure(window: window) == window,
                """
                maxWindowExposure(\(window)) is \
                \(PagedSequenceKV.maxWindowExposure(window: window)), not \(window). Both the \
                contract's CBv2PagedRingGeometry.attendableTokens and the shipping \
                PagedKVPool.ringPageCount read this one function, so they moved TOGETHER and \
                ringFormulaMatchesPagedKVPool still passed. Widening the exposure grows every \
                windowed ring: re-derive the page counts before changing this.
                """)
        }
    }

    /// The contract's formula must always satisfy the contract's own
    /// construction floor, which is the guard `PagedKVPool
    /// .checkedRingPageCount` enforces at pool build. A formula that can
    /// under-run its own floor would make every pool with that geometry
    /// `backendIneligible` instead of merely under-sized.
    @Test func ringAlwaysCoversAttendableRangePlusOneRound() {
        for window in Self.windows {
            for pageSize in Self.pageSizes {
                for chunk in Self.chunks {
                    let tokens =
                        CBv2PagedRingGeometry.ringPageCount(
                            window: window, pageSize: pageSize, maxPrefillChunk: chunk) * pageSize
                    let required = CBv2PagedRingGeometry.requiredTokens(
                        window: window, maxPrefillChunk: chunk)
                    #expect(
                        tokens >= required,
                        """
                        window \(window), pageSize \(pageSize), chunk \(chunk): ring holds \
                        \(tokens) tokens but must cover \(required) (max of attendable \
                        \(CBv2PagedRingGeometry.attendableTokens(window: window)) + span \
                        \(CBv2PagedSpeculation.maxSpeculativeSpan), and one chunk \(chunk))
                        """)
                }
            }
        }
    }

    /// The seam's `attendableTokens` must be exactly what the ROW can be
    /// asked to gather — not a number that happens to match today.
    ///
    /// THIS IS THE CONDITION PIN. `attendableTokens` dropped its
    /// `maxPrefillChunk` term because `PagedSequenceKV.update` and
    /// `PagedLayerCache.prefillKV` gather a chunk's window history BEFORE
    /// writing the chunk, which collapsed `retainedCount` to
    /// `min(written, window)`. Restore a post-write gather anywhere on either
    /// path and `retainedCount` climbs above `window` again — that is what
    /// this measures, on a REAL row driven through a REAL `maxPrefillChunk`
    /// write, not on the arithmetic.
    ///
    /// The failure it stands in front of is not hypothetical. A ring sized
    /// from `window` while the row still gathers after writing asks
    /// `gatherRange` for `window - 1 + chunk` out of `window + span` and
    /// aborts the process. That shipped once.
    @Test(arguments: [(window: 32, chunk: 16), (window: 64, chunk: 64), (window: 128, chunk: 32)])
    func attendableSpanIsWhatARealRowActuallyExposes(_ shape: (window: Int, chunk: Int)) throws {
        let kind = CBv2LayerKind(
            attention: .slidingWindow(shape.window), headDim: 64, kvHeads: 2, queryHeads: 4)
        let cfg = PagedKVPoolConfig(
            capacityBytes: 8 << 20, maxPrefillChunk: shape.chunk,
            nominalMaxSequenceLength: 4096)
        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 2048)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)

        let declared = CBv2PagedRingGeometry.attendableTokens(window: shape.window)
        // Sweep well past the ring so wrap-around is live, ending on a
        // FULL-SIZE chunk — the phase that used to inflate `retainedCount`.
        var written = 0
        while written < 4 * shape.window + 2 * shape.chunk {
            let n = shape.chunk
            _ = row.update(
                keys: MLXArray.zeros([2, n, 64], dtype: .float16),
                values: MLXArray.zeros([2, n, 64], dtype: .float16))
            written += n
            #expect(
                row.retainedCount <= declared,
                """
                window \(shape.window) chunk \(shape.chunk): after a \(n)-token write the row \
                exposes \(row.retainedCount) positions but the seam declares \
                \(declared). CBv2PagedRingGeometry.attendableTokens drops the maxPrefillChunk \
                term ONLY because PagedSequenceKV.update and PagedLayerCache.prefillKV gather \
                the chunk's window history BEFORE writing the chunk. If a post-write gather \
                came back, put the term back FIRST — a ring sized for the window alone aborts \
                the daemon in gatherRange on ordinary prefill.
                """)
        }
        // ...and the row must still be gatherable at that width, which is the
        // half `retainedCount` alone cannot show.
        let (k, _) = row.attendableViews()
        #expect(k.dim(2) == row.retainedCount)
    }

    /// The ROW bound is independent of the window and must be pinned
    /// independently, or a future resize re-derives the ring from the cache
    /// bound alone and hands a long-chunk pool a ring it cannot write into.
    ///
    /// `maxPrefillChunk` used to be implied by the chunk term inside the
    /// attendable span. It is not implied any more.
    @Test func rowBoundSurvivesAChunkThatOutrunsTheWindow() {
        let pageSize = 16
        let span = CBv2PagedSpeculation.maxSpeculativeSpan
        for (window, chunk) in [(128, 2048), (32, 512), (16, 4096)] {
            let required = CBv2PagedRingGeometry.requiredTokens(
                window: window, maxPrefillChunk: chunk)
            let cacheOnly = window + span
            #expect(
                chunk > cacheOnly,
                "fixture is wrong: window \(window) chunk \(chunk) is not row-bound")
            #expect(
                required >= chunk,
                """
                window \(window) chunk \(chunk): the ring must cover ONE maxPrefillChunk write \
                (\(chunk) tokens) and covers \(required). The cache bound alone would give it \
                \(cacheOnly) tokens — \((cacheOnly + pageSize - 1) / pageSize) pages for a \
                \((chunk + pageSize - 1) / pageSize)-page write, so the chunk would lap the \
                ring inside a single bulk-write dispatch with no ordering between the two \
                tokens landing in the same slot.
                """)
            let pages = CBv2PagedRingGeometry.ringPageCount(
                window: window, pageSize: pageSize, maxPrefillChunk: chunk)
            #expect(pages * pageSize >= chunk)
        }
    }

    /// gemma-4's geometry, and the history that makes 65 pages a conclusion
    /// rather than a constant.
    ///
    /// 25 of gemma-4's 30 layers are sliding at window 1,024 with pageSize 16
    /// and the default 512-token prefill chunk. The ring is 65 pages == 1,040
    /// tokens. It was 97 pages == 1,552 tokens, and the 528-token difference
    /// was a measured 1.10x per-sequence KV regression against the contiguous
    /// backend at 10k context.
    ///
    /// 65 was ALSO tried before and reverted, for a reproduced daemon abort.
    /// What changed is stated in `CBv2PagedRingGeometry`'s doc comment and
    /// measured by `attendableSpanIsWhatARealRowActuallyExposes` above; this
    /// test only pins the arithmetic that follows from it, plus the two
    /// things that would have to be true again for 97 to come back.
    @Test func gemma4RingIsSixtyFivePagesBecauseTheRowNoLongerOverExposes() {
        let window = 1024
        let pageSize = 16
        let chunk = 512
        let span = CBv2PagedSpeculation.maxSpeculativeSpan

        #expect(CBv2PagedRingGeometry.attendableTokens(window: window) == 1024)
        #expect(CBv2PagedRingGeometry.requiredTokens(window: window, maxPrefillChunk: chunk) == 1032)
        let landed = CBv2PagedRingGeometry.ringPageCount(
            window: window, pageSize: pageSize, maxPrefillChunk: chunk)
        #expect(landed == 65, "gemma-4's windowed layers ring at 65 pages (1,040 tokens)")
        #expect(landed * pageSize == 1040)

        // Cache-bound, not row-bound, at the production chunk — so the
        // 65 has to survive the row bound being slack here, and
        // `rowBoundSurvivesAChunkThatOutrunsTheWindow` covers the other side.
        #expect(window + span > chunk)

        // What 97 pages WAS: the post-write attendable span. Kept as live
        // arithmetic so the size of the thing that was removed stays visible.
        let postWriteSpan = window - 1 + chunk
        let old = (postWriteSpan + pageSize - 1) / pageSize + (span + pageSize - 1) / pageSize
        #expect(old == 97)
        #expect((old - landed) * pageSize == 512)
        #expect(
            postWriteSpan > landed * pageSize,
            """
            a post-write gather asks for \(postWriteSpan) tokens out of a \(landed * pageSize)-\
            token ring. That inequality IS the reverted abort: it is why the ring may only be \
            this small while both write paths gather BEFORE they write.
            """)
    }
}

// MARK: - Speculative span (PR#86 review, PagedSeamContract.swift:50)

@Suite("CBv2 paged seam contract: speculative span")
struct CBv2PagedSeamContractSpanTests {

    /// The invariant the old `assertSpanCoversMTPBound()` was supposed to
    /// hold and could not: nothing called it, and `assert` is compiled out
    /// under `-O`. This runs in every CI configuration.
    @Test func speculativeSpanCoversMTPDraftBound() {
        let drift = Comment(rawValue: CBv2PagedSpeculation.spanDriftMessage)
        #expect(CBv2PagedSpeculation.spanCoversMTPBound, drift)
        #expect(
            CBv2PagedSpeculation.maxSpeculativeSpan >= CBv2MTPConfig.testedMaxDraftTokens + 1,
            drift)
        // Exercise the shipping validation itself, so the function is not
        // dead again the moment someone deletes its one caller.
        CBv2PagedSpeculation.assertSpanCoversMTPBound()
    }

    /// The validated property must expose the frozen literal unchanged —
    /// otherwise every consumer that reads `maxSpeculativeSpan` (ring sizing,
    /// `speculativeHeadroom`, `supportsSpeculativeWrites`) would be sizing
    /// against something the contract never declared.
    @Test func validatedSpanIsTheDeclaredLiteral() {
        #expect(CBv2PagedSpeculation.maxSpeculativeSpan == CBv2PagedSpeculation.declaredSpan)
        #expect(CBv2PagedSpeculation.maxSpeculativeSpan == 8)
    }

    /// The drift message must name both constants, because the person who
    /// trips it is raising one of them and needs to know which other one to
    /// re-check.
    @Test func driftMessageNamesBothConstants() {
        let message = CBv2PagedSpeculation.spanDriftMessage
        #expect(message.contains("maxSpeculativeSpan"))
        #expect(message.contains("testedMaxDraftTokens"))
        #expect(message.contains("ringPageCount"))
    }
}

// MARK: - Window snapshot boundary (PR#86 review, PagedSeamContract.swift:173)

@Suite("CBv2 paged seam contract: window snapshot boundary")
struct CBv2PagedSeamContractWindowSnapshotTests {

    private static let kvHeads = 2
    private static let headDim = 8

    private func snapshot(base: Int, tokens: Int) -> CBv2PagedWindowSnapshot {
        let shape = [1, Self.kvHeads, tokens, Self.headDim]
        let snapshot = CBv2PagedWindowSnapshot(
            keys: MLXArray.zeros(shape, dtype: .float16),
            values: MLXArray.zeros(shape, dtype: .float16),
            base: base)
        // A well-formed payload must construct; a nil here is a test bug.
        return snapshot!
    }

    /// THE regression. A donation that ended near token 4,096 is indexed by
    /// `PrefixCacheV2` at every whole-block boundary it covers, so a later
    /// lookup can legitimately match at 1,024. The trailing window that row
    /// retained holds absolute positions [3072, 4096); installing it at 1,024
    /// would place those keys at [0, 1024) — silently wrong answers, since
    /// paged storage is indexed by absolute position and nothing downstream
    /// can notice. It must be refused.
    @Test func windowSnapshotIsRefusedAtEveryBoundaryButItsOwn() throws {
        let window = 1024
        let donated = snapshot(base: 3072, tokens: window)
        #expect(donated.endBoundary == 4096)

        // Its own boundary: admissible.
        try donated.requireAdmissible(at: 4096, window: window)

        // The earlier boundary the prefix cache can return: refused, naming
        // both positions.
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                snapshotEnd: 4096, requested: 1024)
        ) {
            try donated.requireAdmissible(at: 1024, window: window)
        }

        // Every other boundary too, including the snapshot's own base and
        // off-by-ones on either side.
        for boundary in [1, 256, 3072, 4095, 4097, 8192] {
            #expect(
                throws: CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                    snapshotEnd: 4096, requested: boundary)
            ) {
                try donated.requireAdmissible(at: boundary, window: window)
            }
        }
    }

    /// A partial window is refused rather than restored. The missing oldest
    /// entries are invisible to attention — they cannot be recovered by a
    /// short replay — so a half window at the right boundary is still a wrong
    /// answer. This is the rule WS-4.2 enforces on the provider side by
    /// requiring every tiling block to be present.
    @Test func partialWindowIsRefusedAtItsOwnBoundary() {
        let window = 1024
        let partial = snapshot(base: 3584, tokens: 512)
        #expect(partial.endBoundary == 4096)
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: 512, required: 1024)
        ) {
            try partial.requireAdmissible(at: 4096, window: window)
        }
    }

    /// An over-long payload is refused for the same reason in reverse: it is
    /// not a window, and installing it would write positions the row's ring
    /// cannot hold at that boundary.
    @Test func overlongWindowIsRefused() {
        let window = 256
        let overlong = snapshot(base: 3072, tokens: 1024)
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: 1024, required: 256)
        ) {
            try overlong.requireAdmissible(at: 4096, window: window)
        }
    }

    /// A boundary shorter than one window: the row's whole history IS its
    /// window, so a snapshot based at 0 covering all of it is exact.
    @Test func historyShorterThanOneWindowIsAWholeWindow() throws {
        let window = 1024
        let short = snapshot(base: 0, tokens: 512)
        try short.requireAdmissible(at: 512, window: window)

        // Still keyed by the boundary: the same payload is not admissible
        // anywhere else, and a hole at the front is still refused.
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                snapshotEnd: 512, requested: 256)
        ) {
            try short.requireAdmissible(at: 256, window: window)
        }
        let holed = snapshot(base: 8, tokens: 504)
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: 504, required: 512)
        ) {
            try holed.requireAdmissible(at: 512, window: window)
        }
    }

    /// Full-attention rows do not come back through this seam at all; their
    /// K/V is restored from `PrefixCacheV2`'s per-layer snapshots. A window
    /// offered to one is refused rather than quietly ignored.
    @Test func fullAttentionRowsHaveNoWindowToRestore() {
        let donated = snapshot(base: 3072, tokens: 1024)
        #expect(throws: CBv2PagedWindowRestoreRefusal.notWindowed(requested: 4096)) {
            try donated.requireAdmissible(at: 4096, window: nil)
        }
        #expect(throws: CBv2PagedWindowRestoreRefusal.notWindowed(requested: 4096)) {
            try donated.requireAdmissible(at: 4096, window: 0)
        }
    }

    /// The extent is read off the payload, never supplied, so a caller cannot
    /// claim a window it does not carry — which is what makes the
    /// wrong-position install unrepresentable rather than merely checked.
    @Test func extentAndBoundaryAreDerivedFromThePayload() {
        for (base, tokens) in [(0, 1), (16, 256), (3072, 1024)] {
            let s = snapshot(base: base, tokens: tokens)
            #expect(s.tokens == tokens)
            #expect(s.keys.dim(2) == tokens)
            #expect(s.endBoundary == base + tokens)
        }
    }

    /// A mis-shaped or corrupt donation yields `nil` instead of trapping: a
    /// cache read must degrade to replay, never abort a multi-tenant daemon.
    @Test func malformedPayloadsAreRefusedWithoutTrapping() {
        let heads = Self.kvHeads
        let dim = Self.headDim
        let ok = MLXArray.zeros([1, heads, 32, dim], dtype: .float16)

        // Negative base.
        #expect(CBv2PagedWindowSnapshot(keys: ok, values: ok, base: -1) == nil)
        // Not 4-D.
        let threeD = MLXArray.zeros([heads, 32, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: threeD, values: threeD, base: 0) == nil)
        // Batched (a window is one row).
        let batched = MLXArray.zeros([2, heads, 32, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: batched, values: batched, base: 0) == nil)
        // Keys and values disagreeing on the token extent.
        let shorter = MLXArray.zeros([1, heads, 16, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: ok, values: shorter, base: 0) == nil)
        // Keys and values disagreeing on head geometry.
        let wideHead = MLXArray.zeros([1, heads, 32, dim * 2], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: ok, values: wideHead, base: 0) == nil)
        // Empty window.
        let empty = MLXArray.zeros([1, heads, 0, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: empty, values: empty, base: 0) == nil)
    }
}
