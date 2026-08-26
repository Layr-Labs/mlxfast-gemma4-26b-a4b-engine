// CBv2PagedFrozenChunkGatherTests.swift
//
// `PagedLayerCache.prefillKVWritingChunk`'s FROZEN-REPLAY branch: during
// prefix reuse's `.frozenFullReplay`, a frozen full row attends the CACHED
// keys for the whole chunk — diagonal included — instead of the freshly
// projected ones it was handed.
//
// WHY THIS EXISTS. A frozen replay re-runs `[C, M)` through the model so the
// SLIDING layers rebuild their windows. Those rows have no windows yet, so
// every projection the replay produces below the dependency-cone frontier is
// POISONED. `CBv2FrozenReplayFullSequenceKV.update` (contiguous) throws the
// replayed projections away and returns `keys[0 ..< absoluteOffset]`, so a
// contiguous frozen row contributes exact keys for every position including
// the one the query sits on. Paged assembled `gather(history) ++ chunk` with
// the chunk half being those poisoned projections, so a paged frozen row
// contributed exact keys BEFORE the current chunk and poisoned ones INSIDE
// it. A position was therefore exact only once the chunk it sits in STARTED
// at or after the cone frontier — which WAS the `+ maxPrefillChunk` (capped
// to `+ maxWindow`) that `PagedKVBackend.requiredFrozenReplayTokens` demanded
// on top of the model-shape bound `cbv2RequiredRecompute`, and which cost
// gemma-4 36% of its measured prefix-reuse saving.
//
// That term is GONE. `derive` grants `windowCount * maxWindow` for paged and
// contiguous alike, `requiredFrozenReplayTokens` and `replayChunkCeiling` are
// deleted, and the adoption guard calls the shared `cbv2RequiredRecompute`.
// This suite is what made that safe to remove, and it is what keeps it safe:
// the frozen branch pinned here is now load-bearing FOR THE BOUND, so pulling
// the branch fails every frozen test below rather than quietly costing
// exactness one chunk at a time.
//
// The bytes were always there; the layer was simply not reading them.
//
// THREE MECHANISM FACTS these tests also pin, because the fix is only safe
// while all three hold:
//   1. `PagedSequenceKV.adoptFrozen` refuses windowed rows, so a frozen row
//      has `ringPages == nil` — no ring, nothing recycled, no eviction
//      branch in `gatherRange` to trip.
//   2. `PagedSequenceKV.write` is CURSOR-ONLY below M: it advances
//      `absoluteOffset` and touches no storage. That is what makes the
//      gather legal AFTER the write (`gatherRange` bounds at the cursor,
//      which by then is `queryStart + chunk`) and what keeps the WS-1.2
//      pre-write gather intact for every non-frozen — i.e. every windowed —
//      row. Ring geometry is untouched by this file.
//   3. A frozen chunk never straddles M, so the range read is wholly cached.
//
// Everything runs on synthetic K/V through the real `PagedKVBackend`. The
// poison is `codedKV` with a different tag: numerically loud at every single
// element, so `arrayEqual` (never `allClose`) is the right assertion — a
// diagonal of 16 poisoned keys out of 48 is not a tolerance question.

import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedFrozenChunkGather")
struct CBv2PagedFrozenChunkGatherTests {

    // MARK: - Fixture

    private let headDim = 64
    private let kvHeads = 2
    private let queryHeads = 4
    private let maxLength = 1024

    private var scale: Float { 1.0 / Float(headDim).squareRoot() }

    /// Frozen adoption is full-attention only, so every layer here is full.
    private var fullLayer: CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, sharesKVWithLayer: nil,
            headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func config(maxPrefillChunk: Int) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: 256 << 20, dtype: .float32,
            maxPrefillChunk: maxPrefillChunk, nominalMaxSequenceLength: maxLength)
    }

    /// `[kvHeads, n, headDim]` in which both the absolute position and `tag`
    /// are encoded in every element, so two tags differ at every element of
    /// every position. Cached K/V and the replay's poisoned projections carry
    /// different tags: reading the wrong one is loud, not marginal.
    private func codedKV(_ positions: Range<Int>, tag: Int) -> (MLXArray, MLXArray) {
        let n = positions.count
        var kflat = [Float](repeating: 0, count: kvHeads * n * headDim)
        var vflat = [Float](repeating: 0, count: kvHeads * n * headDim)
        var i = 0
        let t = Float(tag)
        for h in 0 ..< kvHeads {
            for p in positions {
                for d in 0 ..< headDim {
                    kflat[i] = Float(p % 89) * 0.011 + Float(d) * 0.001 + Float(h) * 0.1 + t * 0.37
                    vflat[i] =
                        Float(p % 89) * 0.021 - Float(d) * 0.0005 + Float(h) * 0.05 - t * 0.29
                    i += 1
                }
            }
        }
        return (
            MLXArray(kflat, [kvHeads, n, headDim]), MLXArray(vflat, [kvHeads, n, headDim])
        )
    }

    private func queryBlock(_ count: Int, seed: UInt64) -> MLXArray {
        MLXRandom.normal([1, queryHeads, count, headDim], key: MLXRandom.key(seed))
    }

    private func assertIdentical(
        _ got: MLXArray, _ want: MLXArray, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, "\(what): shape", sourceLocation: sourceLocation)
        guard got.shape == want.shape else { return }
        let delta = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
        #expect(
            arrayEqual(got, want).item(Bool.self),
            "\(what) are not bit-identical: max |delta| = \(delta)",
            sourceLocation: sourceLocation)
    }

    /// A fresh row holding `cached[0 ..< upTo]` as ORDINARY storage: the cold
    /// twin of a frozen row whose cursor sits at `upTo`.
    private func coldRow(
        _ backend: PagedKVBackend, cached: (MLXArray, MLXArray), upTo: Int
    ) throws -> (state: [CBv2SequenceKV?], row: PagedSequenceKV) {
        let state = try backend.makeSequenceState(
            layerKinds: [fullLayer], promptLength: maxLength, maxLength: maxLength)
        let row = try #require(state[0] as? PagedSequenceKV)
        if upTo > 0 {
            row.write(
                keys: cached.0[0..., 0 ..< upTo, 0...],
                values: cached.1[0..., 0 ..< upTo, 0...])
        }
        return (state, row)
    }

    /// A fresh row with `[0, M)` adopted frozen and the cursor rewound to C.
    private func frozenRow(
        _ backend: PagedKVBackend, cached: (MLXArray, MLXArray), replayStart: Int
    ) throws -> (state: [CBv2SequenceKV?], row: PagedSequenceKV) {
        let state = try backend.makeSequenceState(
            layerKinds: [fullLayer], promptLength: maxLength, maxLength: maxLength)
        let row = try #require(state[0] as? PagedSequenceKV)
        row.adoptFrozen(keys: cached.0, values: cached.1, replayStart: replayStart)
        return (state, row)
    }

    /// One prompt chunk through a FRESH layer cache — "run alone" has to mean
    /// alone, not "bound to a cache that already saw another row".
    private func attend(
        _ backend: PagedKVBackend, row: PagedSequenceKV,
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) -> MLXArray {
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([row])
        return cache.updateAndAttend(
            queries: queries,
            keys: keys.expandedDimensions(axis: 0),
            values: values.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)
    }

    // MARK: - 1. The frozen chunk reads its own cached diagonal

    /// The headline. A frozen row mid-replay is handed POISONED projections
    /// for its chunk and must still produce the answer a cold twin holding
    /// the exact K/V produces — bit for bit, at the FIRST replayed chunk,
    /// with no slack of any kind.
    ///
    /// That is the whole content of the `+ maxWindow` term: it existed only
    /// to push every query past the chunk whose diagonal was poisoned. Once
    /// the diagonal is cached, "the chunk must START at or after the cone
    /// frontier" collapses to contiguous's "the POSITION must be at or after
    /// the cone frontier", and the extra window buys nothing.
    @Test func frozenReplayChunkAttendsCachedKeysNotReplayedProjections() throws {
        let m = 64
        let c = 32
        let chunk = 16
        let backend = try PagedKVBackend(
            layerKinds: [fullLayer], config: config(maxPrefillChunk: chunk))
        let cached = codedKV(0 ..< m, tag: 1)

        let frozen = try frozenRow(backend, cached: cached, replayStart: c)
        let cold = try coldRow(backend, cached: cached, upTo: c)
        let control = try coldRow(backend, cached: cached, upTo: c)
        defer { for s in [frozen.state, cold.state, control.state] { backend.release(s) } }

        // Mechanism facts 1 and 2, asserted rather than assumed.
        #expect(frozen.row.frozenHighWater == m)
        #expect(frozen.row.absoluteOffset == c, "the cursor reports C while storage covers M")
        #expect(frozen.row.windowSize == nil, "adoptFrozen refuses windowed rows")
        #expect(cold.row.frozenHighWater == 0)
        #expect(cold.row.absoluteOffset == c)

        let q = queryBlock(chunk, seed: 0xF0_0D_01)
        // What a replay ACTUALLY hands the layer below the cone frontier.
        let poison = codedKV(c ..< (c + chunk), tag: 9)
        let exact = (
            cached.0[0..., c ..< (c + chunk), 0...], cached.1[0..., c ..< (c + chunk), 0...]
        )

        let got = attend(backend, row: frozen.row, queries: q, keys: poison.0, values: poison.1)
        let want = attend(backend, row: cold.row, queries: q, keys: exact.0, values: exact.1)
        assertIdentical(got, want, "frozen replay chunk vs a cold twin holding the same K/V")

        // Teeth: the poison is genuinely different, so the equality above is
        // a property of the gather and not of the fixture.
        let poisoned = attend(
            backend, row: control.row, queries: q, keys: poison.0, values: poison.1)
        #expect(
            !arrayEqual(poisoned, want).item(Bool.self),
            """
            premise: attending the replayed projections must change the answer, \
            or this test proves nothing
            """)

        // The cursor advanced and the frozen storage is still byte-exact —
        // the write below M wrote nothing (mechanism fact 2).
        #expect(frozen.row.absoluteOffset == c + chunk)
        let (storedK, storedV) = frozen.row.gatherRange(start: 0, count: c + chunk)
        assertIdentical(
            storedK, cached.0[0..., 0 ..< (c + chunk), 0...].expandedDimensions(axis: 0),
            "frozen keys after a replayed chunk")
        assertIdentical(
            storedV, cached.1[0..., 0 ..< (c + chunk), 0...].expandedDimensions(axis: 0),
            "frozen values after a replayed chunk")
    }

    /// The same property swept across where the chunk sits and how long it
    /// is, including a chunk that starts at the replay start itself (no
    /// history at all) and one that ends exactly at M. Exactness must not
    /// depend on the chunk's alignment — that dependence IS the term being
    /// removed.
    @Test func frozenReplayIsExactAtEveryChunkAlignment() throws {
        let m = 96
        let cases: [(replayStart: Int, chunk: Int)] = [
            (0, 16),  // no history: the chunk is the whole view
            (16, 8),  // chunk smaller than a page
            (32, 16),  // page aligned
            (40, 24),  // straddles pages on both ends
            (80, 16),  // ends exactly at M
        ]
        for (index, testCase) in cases.enumerated() {
            let backend = try PagedKVBackend(
                layerKinds: [fullLayer], config: config(maxPrefillChunk: testCase.chunk))
            let cached = codedKV(0 ..< m, tag: 2)
            let frozen = try frozenRow(backend, cached: cached, replayStart: testCase.replayStart)
            let cold = try coldRow(backend, cached: cached, upTo: testCase.replayStart)
            defer { for s in [frozen.state, cold.state] { backend.release(s) } }

            let end = testCase.replayStart + testCase.chunk
            #expect(end <= m, "premise: a frozen chunk never straddles M")
            let q = queryBlock(testCase.chunk, seed: 0xA11_00 + UInt64(index))
            let poison = codedKV(testCase.replayStart ..< end, tag: 30 + index)
            let exact = (
                cached.0[0..., testCase.replayStart ..< end, 0...],
                cached.1[0..., testCase.replayStart ..< end, 0...]
            )

            let got = attend(
                backend, row: frozen.row, queries: q, keys: poison.0, values: poison.1)
            let want = attend(backend, row: cold.row, queries: q, keys: exact.0, values: exact.1)
            assertIdentical(
                got, want,
                "frozen chunk [\(testCase.replayStart), \(end)) of M = \(m)")
        }
    }

    /// The blocked path (WS-0.2p). A chunk wider than `queryBlockSize` is
    /// attended in blocks that SLICE the one hoisted gather; the frozen view
    /// is longer than the ordinary one by exactly the chunk, so a block-bound
    /// computed from the wrong `historyCount` would land here.
    @Test func frozenReplayIsExactWhenTheChunkIsQueryBlocked() throws {
        let chunk = 192
        let m = 512
        let c = 256
        #expect(
            CBv2AttentionV1.shouldBlockQueries(chunk) || CBv2AttentionV1.queryBlockSize == 0,
            "premise: a \(chunk)-token chunk is blocked unless the kill switch is set")

        let backend = try PagedKVBackend(
            layerKinds: [fullLayer], config: config(maxPrefillChunk: chunk))
        let cached = codedKV(0 ..< m, tag: 3)
        let frozen = try frozenRow(backend, cached: cached, replayStart: c)
        let cold = try coldRow(backend, cached: cached, upTo: c)
        defer { for s in [frozen.state, cold.state] { backend.release(s) } }

        let q = queryBlock(chunk, seed: 0xB10C_C0DE)
        let poison = codedKV(c ..< (c + chunk), tag: 11)
        let exact = (
            cached.0[0..., c ..< (c + chunk), 0...], cached.1[0..., c ..< (c + chunk), 0...]
        )
        let got = attend(backend, row: frozen.row, queries: q, keys: poison.0, values: poison.1)
        let want = attend(backend, row: cold.row, queries: q, keys: exact.0, values: exact.1)
        assertIdentical(got, want, "query-blocked frozen chunk vs its cold twin")
    }

    // MARK: - 2. The ordinary path is untouched

    /// `frozenHighWater <= absoluteOffset` must take exactly the pre-existing
    /// code path: gather the history BEFORE the write, splice on the chunk
    /// tensor the layer was HANDED, and write it.
    ///
    /// The oracle is built here rather than borrowed from the cache, so this
    /// pins the ordinary path against an independent computation: one SDPA
    /// over `gather(history) ++ chunk` with the absolute-coordinate BOOL mask
    /// this file pins (`kpos .<= qpos`, always `.array`).
    @Test func ordinaryPrefillStillAttendsTheProjectionsItWasHanded() throws {
        let history = 40
        let chunk = 24
        #expect(
            !CBv2AttentionV1.shouldBlockQueries(chunk),
            "premise: this chunk takes the single unblocked SDPA call")

        let backend = try PagedKVBackend(
            layerKinds: [fullLayer], config: config(maxPrefillChunk: chunk))
        let prior = codedKV(0 ..< history, tag: 1)
        let fresh = codedKV(history ..< (history + chunk), tag: 5)
        let subject = try coldRow(backend, cached: prior, upTo: history)
        defer { backend.release(subject.state) }
        #expect(subject.row.frozenHighWater == 0, "premise: an ordinary row is never frozen")

        let (historyKeys, historyValues) = subject.row.gatherRange(start: 0, count: history)
        eval(historyKeys, historyValues)

        let q = queryBlock(chunk, seed: 0x0D_D1_7E)
        let keys = concatenated([historyKeys, fresh.0.expandedDimensions(axis: 0)], axis: 2)
        let values = concatenated([historyValues, fresh.1.expandedDimensions(axis: 0)], axis: 2)
        let qpos = MLXArray(Int32(history) ..< Int32(history + chunk)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(history + chunk)).expandedDimensions(axis: 0)
        let want = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale,
            mask: .array(kpos .<= qpos), sinks: nil)

        let got = attend(backend, row: subject.row, queries: q, keys: fresh.0, values: fresh.1)
        assertIdentical(got, want, "ordinary prefill vs gather(history) ++ the handed chunk")

        // And the chunk was actually WRITTEN — a frozen-branch slip would
        // leave the row's storage short.
        #expect(subject.row.absoluteOffset == history + chunk)
        let (storedK, storedV) = subject.row.gatherRange(start: history, count: chunk)
        assertIdentical(storedK, fresh.0.expandedDimensions(axis: 0), "written chunk keys")
        assertIdentical(storedV, fresh.1.expandedDimensions(axis: 0), "written chunk values")
    }

    /// The boundary itself: a row adopted frozen with `C == M` (prefix
    /// reuse's zero-replay restore form) sits AT the high-water, so
    /// `frozenHighWater > absoluteOffset` is false and its next chunk is an
    /// ordinary appending prefill of `[M, ...)`.
    ///
    /// Getting this wrong is not a wrong number, it is a trap: the frozen
    /// branch would ask `gatherRange` for positions at and beyond M that were
    /// never written.
    @Test func aRowAtItsFrozenHighWaterTakesTheOrdinaryPath() throws {
        let m = 48
        let chunk = 16
        let backend = try PagedKVBackend(
            layerKinds: [fullLayer], config: config(maxPrefillChunk: chunk))
        let cached = codedKV(0 ..< m, tag: 4)
        let atBoundary = try frozenRow(backend, cached: cached, replayStart: m)
        let twin = try coldRow(backend, cached: cached, upTo: m)
        defer { for s in [atBoundary.state, twin.state] { backend.release(s) } }

        #expect(atBoundary.row.frozenHighWater == m)
        #expect(atBoundary.row.absoluteOffset == m, "the restore form parks both cursors at M")

        let fresh = codedKV(m ..< (m + chunk), tag: 6)
        let q = queryBlock(chunk, seed: 0xBEEF_0001)
        let got = attend(backend, row: atBoundary.row, queries: q, keys: fresh.0, values: fresh.1)
        let want = attend(backend, row: twin.row, queries: q, keys: fresh.0, values: fresh.1)
        assertIdentical(got, want, "append above M vs an unfrozen twin holding the same prefix")

        // Above M the write is a real one, on both rows.
        for (label, row) in [("frozen-at-M", atBoundary.row), ("twin", twin.row)] {
            #expect(row.absoluteOffset == m + chunk)
            let (storedK, _) = row.gatherRange(start: m, count: chunk)
            assertIdentical(storedK, fresh.0.expandedDimensions(axis: 0), "\(label) wrote [M, M+n)")
        }
    }

    // MARK: - 3. Packed

    /// `B > 1`, every row frozen, every row at a DIFFERENT replay start with
    /// a DIFFERENT cached prefix. Each must gather its OWN frozen chunk.
    ///
    /// The tags make the two ways to get this wrong loud: gathering row 0's
    /// range for every row, and gathering each row's range at row 0's
    /// coordinates. Both would land on real numbers from a real batchmate,
    /// which is why this is `arrayEqual` against per-row cold twins rather
    /// than a shape or a tolerance check.
    @Test func packedFrozenReplayGathersEachRowsOwnFrozenChunk() throws {
        let m = 96
        let chunk = 16
        let starts = [32, 48, 64]
        let backend = try PagedKVBackend(
            layerKinds: [fullLayer], config: config(maxPrefillChunk: chunk))

        var states: [[CBv2SequenceKV?]] = []
        var packedRows: [PagedSequenceKV] = []
        var soloRows: [PagedSequenceKV] = []
        var coldRows: [PagedSequenceKV] = []
        var queries: [MLXArray] = []
        var poisonKeys: [MLXArray] = []
        var poisonValues: [MLXArray] = []
        var exactKeys: [MLXArray] = []
        var exactValues: [MLXArray] = []
        for (index, start) in starts.enumerated() {
            let cached = codedKV(0 ..< m, tag: 1 + index)
            let packed = try frozenRow(backend, cached: cached, replayStart: start)
            let solo = try frozenRow(backend, cached: cached, replayStart: start)
            let cold = try coldRow(backend, cached: cached, upTo: start)
            states.append(contentsOf: [packed.state, solo.state, cold.state])
            packedRows.append(packed.row)
            soloRows.append(solo.row)
            coldRows.append(cold.row)

            let poison = codedKV(start ..< (start + chunk), tag: 40 + index)
            poisonKeys.append(poison.0.expandedDimensions(axis: 0))
            poisonValues.append(poison.1.expandedDimensions(axis: 0))
            exactKeys.append(cached.0[0..., start ..< (start + chunk), 0...])
            exactValues.append(cached.1[0..., start ..< (start + chunk), 0...])
            queries.append(queryBlock(chunk, seed: 0x9AC_0 + UInt64(index)))
        }
        defer { for s in states { backend.release(s) } }

        let packedCache = backend.makeLayerCaches()[0]
        packedCache.setRows(packedRows)
        let packed = packedCache.updateAndAttend(
            queries: concatenated(queries, axis: 0),
            keys: concatenated(poisonKeys, axis: 0),
            values: concatenated(poisonValues, axis: 0),
            scale: scale, sinks: nil)
        #expect(packed.shape == [starts.count, queryHeads, chunk, headDim])

        for index in starts.indices {
            // Exactness: the packed frozen row equals a cold twin that holds
            // this row's own cached K/V.
            let want = attend(
                backend, row: coldRows[index], queries: queries[index],
                keys: exactKeys[index], values: exactValues[index])
            assertIdentical(
                packed[index ..< (index + 1)], want,
                "packed frozen row \(index) (C = \(starts[index])) vs its cold twin")

            // Independence: and equals the same frozen row run alone.
            let alone = attend(
                backend, row: soloRows[index], queries: queries[index],
                keys: poisonKeys[index].squeezed(axis: 0),
                values: poisonValues[index].squeezed(axis: 0))
            assertIdentical(
                packed[index ..< (index + 1)], alone,
                "packed frozen row \(index) vs the same row run alone")

            #expect(packedRows[index].absoluteOffset == starts[index] + chunk)
        }
    }

    /// A packed group where only SOME rows are frozen — the realistic mixed
    /// batch, since `EngineLoopV2` groups prompt rows by chunk length, not by
    /// whether they are replaying. The frozen rows must read their pages and
    /// the ordinary rows must attend what they were handed, in one call.
    @Test func packedMixOfFrozenAndOrdinaryRowsTakesBothPathsAtOnce() throws {
        let m = 64
        let start = 32
        let chunk = 16
        let backend = try PagedKVBackend(
            layerKinds: [fullLayer], config: config(maxPrefillChunk: chunk))

        let frozenCached = codedKV(0 ..< m, tag: 1)
        let plainPrior = codedKV(0 ..< start, tag: 2)
        let frozen = try frozenRow(backend, cached: frozenCached, replayStart: start)
        let frozenTwin = try coldRow(backend, cached: frozenCached, upTo: start)
        let plain = try coldRow(backend, cached: plainPrior, upTo: start)
        let plainTwin = try coldRow(backend, cached: plainPrior, upTo: start)
        defer {
            for s in [frozen.state, frozenTwin.state, plain.state, plainTwin.state] {
                backend.release(s)
            }
        }

        let poison = codedKV(start ..< (start + chunk), tag: 50)
        let plainFresh = codedKV(start ..< (start + chunk), tag: 51)
        let frozenExact = (
            frozenCached.0[0..., start ..< (start + chunk), 0...],
            frozenCached.1[0..., start ..< (start + chunk), 0...]
        )
        let q0 = queryBlock(chunk, seed: 0xC0DE_A000)
        let q1 = queryBlock(chunk, seed: 0xC0DE_A001)

        let cache = backend.makeLayerCaches()[0]
        cache.setRows([frozen.row, plain.row])
        let packed = cache.updateAndAttend(
            queries: concatenated([q0, q1], axis: 0),
            keys: concatenated(
                [poison.0.expandedDimensions(axis: 0), plainFresh.0.expandedDimensions(axis: 0)],
                axis: 0),
            values: concatenated(
                [poison.1.expandedDimensions(axis: 0), plainFresh.1.expandedDimensions(axis: 0)],
                axis: 0),
            scale: scale, sinks: nil)

        // Row 0 is frozen: it must have read its cached diagonal.
        assertIdentical(
            packed[0 ..< 1],
            attend(
                backend, row: frozenTwin.row, queries: q0,
                keys: frozenExact.0, values: frozenExact.1),
            "packed frozen row vs a cold twin holding the same cached K/V")
        // Row 1 is ordinary: it must have attended what it was handed.
        assertIdentical(
            packed[1 ..< 2],
            attend(
                backend, row: plainTwin.row, queries: q1,
                keys: plainFresh.0, values: plainFresh.1),
            "packed ordinary row vs the same row run alone")

        #expect(frozen.row.absoluteOffset == start + chunk)
        #expect(plain.row.absoluteOffset == start + chunk)
        // The frozen row wrote nothing; the ordinary row wrote its chunk.
        let (frozenStored, _) = frozen.row.gatherRange(start: start, count: chunk)
        assertIdentical(
            frozenStored, frozenExact.0.expandedDimensions(axis: 0),
            "the frozen row's storage is still the adopted prefix")
        let (plainStored, _) = plain.row.gatherRange(start: start, count: chunk)
        assertIdentical(
            plainStored, plainFresh.0.expandedDimensions(axis: 0),
            "the ordinary row wrote the chunk it was handed")
    }

    // MARK: - 4. The bound this unblocks

    /// The consequence, on the real model fixture: a paged frozen replay run
    /// at the CONTIGUOUS replay bound — `cbv2RequiredRecompute` with no
    /// `+ maxPrefillChunk`/`+ maxWindow` slack on top — is token-exact
    /// against a cold twin.
    ///
    /// This was the evidence for the narrowing, and the order mattered: a
    /// bound narrowed before the layer can honour it is a silent wrong
    /// answer, so `CBv2PrefixReuseCapability.derive` was equalized AFTER this
    /// passed, not before. It has since landed — paged grants exactly
    /// `cbv2RequiredRecompute` — so this is now the regression guard rather
    /// than the justification for one.
    ///
    /// The state is still built by hand rather than through
    /// `plan(matchedBoundary:)`, and deliberately so: R is pinned to the
    /// model-shape cone bound, never to whatever the capability happens to
    /// grant. Reading the grant here would make the test pass by construction
    /// the moment anyone re-widens it.
    ///
    /// It does NOT carry the mechanism proof — `[full, sliding, sliding,
    /// full]` has no frozen full layer with an INEXACT input (layer 0 reads
    /// the exact embedding, layer 3 feeds only discarded pre-M logits), so
    /// the poisoned diagonal is unobservable here. That proof is the
    /// layer-level suite above, which fails on every frozen case the moment
    /// the branch is removed. This one is the regression guard that keeps
    /// R == `cbv2RequiredRecompute` honest end to end.
    @Test func frozenReplayAtTheContiguousBoundIsTokenExact() throws {
        let window = 16
        let chunkSize = 8
        let matched = 72
        let steps = 3 * window
        let model = TinyTestModel.make(
            seed: 0xD00D_F00D, headDim: 64, stackedSlidingFull: true, windowSize: window)
        let kinds = model.layerKinds
        #expect(kinds.count == 4, "shape must be [full, sliding, sliding, full]")
        let prompt = makePromptTokens(length: 73, seed: 0x5EED)

        // The model-shape cone bound. Paged now grants exactly this, same as
        // contiguous; R is pinned to the bound itself, never to the grant.
        let bound = cbv2RequiredRecompute(layerKinds: kinds, matched: matched)
        #expect(bound == 2 * window)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        #expect(
            capability.conservativeReplayBoundTokens == bound,
            "paged grants the contiguous bound — no extra window")
        #expect(
            capability.conservativeReplayBoundTokens
                == CBv2PrefixReuseCapability.derive(
                    layerKinds: kinds, backend: .contiguousUnquantized
                ).conservativeReplayBoundTokens,
            """
            the two backends' replay bounds are ONE number — a paged-specific slack \
            term is a regression, not a tuning knob
            """)
        let replayStart = matched - bound

        func makeBackend() throws -> PagedKVBackend {
            try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 64 << 20, maxPrefillChunk: chunkSize,
                    nominalMaxSequenceLength: 256))
        }

        /// Prefill `[from, upTo)` in `chunkSize` pieces, never straddling M,
        /// then greedy-decode. Returns the generated tokens AND every sliding
        /// row's retained window, evaluated before the state is released.
        ///
        /// The windows are the sharp comparison. Greedy tokens are the
        /// user-visible property but a weak instrument on a fixture this
        /// small — a replay well below the dependency cone still agrees token
        /// for token here. The retained K/V does not: it differs the moment
        /// any window is rebuilt from a short history.
        func run(
            backend: PagedKVBackend, state: [CBv2SequenceKV?], from: Int, decode: Int
        ) -> (tokens: [Int], windows: [Int: (keys: MLXArray, values: MLXArray)]) {
            let caches: [CBv2AttendingLayerCache] = backend.makeLayerCaches()
            for (i, kind) in kinds.enumerated() where kind.sharesKVWithLayer == nil {
                caches[i].setRows([state[i]!])
            }
            var index = from
            let upTo = prompt.count - 1
            while index < upTo {
                var count = min(chunkSize, upTo - index)
                if index < matched { count = min(count, matched - index) }
                let slice = Array(prompt[index ..< (index + count)])
                _ = model.forward(
                    tokens: MLXArray(slice.map(Int32.init)).reshaped(1, slice.count),
                    caches: caches)
                index += count
            }
            // Captured HERE, at the end of prefill — this is the window the
            // replay rebuilt. Snapshotting after the decode loop instead
            // would compare only positions generated long past the cone and
            // could not tell a short replay from a sufficient one.
            var windows: [Int: (keys: MLXArray, values: MLXArray)] = [:]
            for (i, kind) in kinds.enumerated() where kind.attention != .full {
                let snapshot = state[i]!.snapshot()
                eval(snapshot.keys, snapshot.values)
                windows[i] = (snapshot.keys, snapshot.values)
            }
            var current = prompt.last!
            var generated: [Int] = []
            for _ in 0 ..< decode {
                let logits = model.forward(
                    tokens: MLXArray([Int32(current)]).reshaped(1, 1), caches: caches)
                current = Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
                generated.append(current)
            }
            return (generated, windows)
        }

        // Cold arm.
        let coldBackend = try makeBackend()
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: 256)
        let cold = run(backend: coldBackend, state: coldState, from: 0, decode: steps)
        coldBackend.release(coldState)

        // Donor arm: a cold prefill of [0, M), full rows snapshotted exactly
        // as a prefix cache holds them. Paged snapshots are lazy views over
        // the shared slabs, so they must be evaluated before the donor's
        // pages are recycled.
        let backend = try makeBackend()
        let donorState = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: matched, maxLength: 256)
        let donorCaches: [CBv2AttendingLayerCache] = backend.makeLayerCaches()
        for (i, kind) in kinds.enumerated() where kind.sharesKVWithLayer == nil {
            donorCaches[i].setRows([donorState[i]!])
        }
        var index = 0
        while index < matched {
            let count = min(chunkSize, matched - index)
            let slice = Array(prompt[index ..< (index + count)])
            _ = model.forward(
                tokens: MLXArray(slice.map(Int32.init)).reshaped(1, slice.count),
                caches: donorCaches)
            index += count
        }
        var donated: [Int: (keys: MLXArray, values: MLXArray)] = [:]
        for (i, kind) in kinds.enumerated() where kind.attention == .full {
            let snapshot = donorState[i]!.snapshot()
            eval(snapshot.keys, snapshot.values)
            #expect(snapshot.offset == matched)
            donated[i] = (snapshot.keys.squeezed(axis: 0), snapshot.values.squeezed(axis: 0))
        }
        backend.release(donorState)

        // Adopt arm: full rows frozen through M with the cursor at C, sliding
        // rows empty at C. The dual-cursor `.frozenFullReplay` shape, with
        // R == the CONTIGUOUS bound.
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: 256)
        for (i, kind) in kinds.enumerated() {
            guard let row = state[i] as? PagedSequenceKV else { continue }
            if let full = donated[i] {
                row.adoptFrozen(keys: full.keys, values: full.values, replayStart: replayStart)
                #expect(row.frozenHighWater == matched)
                #expect(row.absoluteOffset == replayStart)
            } else {
                #expect(kind.attention == .slidingWindow(window))
                row.fastForward(to: replayStart)
                #expect(row.baseOffset == replayStart)
            }
        }
        let adopted = run(backend: backend, state: state, from: replayStart, decode: steps)
        backend.release(state)

        let agreed = zip(adopted.tokens, cold.tokens).prefix(while: { $0.0 == $0.1 }).count
        #expect(
            adopted.tokens == cold.tokens,
            """
            a frozen replay of R = \(bound) (the contiguous bound, C = \(replayStart), \
            M = \(matched)) diverged from a cold twin after \(agreed) of \(steps) tokens
            """)
        #expect(!cold.windows.isEmpty, "premise: the layout has sliding rows to rebuild")
        for (layer, want) in cold.windows {
            let got = try #require(adopted.windows[layer])
            assertIdentical(
                got.keys, want.keys, "layer \(layer) rebuilt window keys at R = \(bound)")
            assertIdentical(
                got.values, want.values, "layer \(layer) rebuilt window values at R = \(bound)")
        }
    }
}
