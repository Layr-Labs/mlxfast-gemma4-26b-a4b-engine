// CBv2PagedBackendTests.swift
//
// WS-C unit tests: paged pool bookkeeping, per-sequence page tables,
// windowed rings, reservation-based admission, and backend lifecycle.
// No model weights required.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

// MARK: - Track P poison-page surface (forward declaration, see T4 below)

/// Out-of-band answers meaning "track P's poison-page surface is not present".
private let poisonPageAbsent: Int32 = -1
private let usablePageCountAbsent = -1

/// The three members track P adds to `PagedKVPool` in WS-1.1, declared here as
/// protocol requirements WITH SENTINEL DEFAULTS.
///
/// The concrete methods win witness matching the moment they exist on
/// `PagedKVPool`; until then the defaults answer and the poison-page tests
/// fail with an explicit "not landed" message. This is purely a COMPILE-ORDER
/// device — several agents share this test target, and a hard reference to a
/// symbol that has not landed would break everyone's build rather than fail
/// one test. It weakens nothing: the assertions are the real invariant on both
/// sides of the change, and a signature that does not match exactly falls back
/// to the sentinel and stays red rather than passing vacuously.
private protocol CBv2PoisonPageProviding: AnyObject {
    func poisonPage(group key: PagedKVGroupKey) -> Int32
    func usablePageCount(group key: PagedKVGroupKey) -> Int
    func isAllocatablePage(_ page: Int32, group key: PagedKVGroupKey) -> Bool
}

extension CBv2PoisonPageProviding {
    func poisonPage(group key: PagedKVGroupKey) -> Int32 { poisonPageAbsent }
    func usablePageCount(group key: PagedKVGroupKey) -> Int { usablePageCountAbsent }
    func isAllocatablePage(_ page: Int32, group key: PagedKVGroupKey) -> Bool { true }
}

extension PagedKVPool: CBv2PoisonPageProviding {}

@Suite("CBv2PagedBackend")
struct CBv2PagedBackendTests {

    // MARK: - Helpers

    private func fullKind(
        headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func windowedKind(
        _ window: Int, headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
            queryHeads: queryHeads)
    }

    private func config(
        capacityBytes: Int = 8 << 20, maxPrefillChunk: Int = 64,
        nominalMaxLen: Int = 1024
    ) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: capacityBytes, maxPrefillChunk: maxPrefillChunk,
            nominalMaxSequenceLength: nominalMaxLen)
    }

    private func randomKV(heads: Int, tokens: Int, dim: Int) -> MLXArray {
        MLXRandom.normal([heads, tokens, dim], dtype: .float16)
    }

    private func assertEqualArrays(
        _ a: MLXArray, _ b: MLXArray,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(a.shape == b.shape, sourceLocation: sourceLocation)
        #expect(
            arrayEqual(a, b).item(Bool.self),
            "arrays differ",
            sourceLocation: sourceLocation)
    }

    // MARK: - Pool

    @Test func poolGroupsAndAccounting() throws {
        let kinds = [fullKind(), fullKind(), windowedKind(32)]
        let pool = try PagedKVPool(layerKinds: kinds, config: config())
        // One group: all layers share (kvHeads=2, headDim=64).
        #expect(pool.groupKeys == [PagedKVGroupKey(kvHeads: 2, headDim: 64)])
        #expect(pool.bytesInUse == 0)
        #expect(pool.bytesReserved == 0)
        #expect(pool.bytesCapacity > 0)
        #expect(pool.bytesCapacity <= 8 << 20)
    }

    // `poolRejectsQuantSchemes` lived here. It asserted that
    // `PagedKVPool.init` throws `backendIneligible` for a non-fp16
    // `CBv2KVQuantScheme`. KV quantization is retired from the product and
    // the enum is gone, so a quantized pool config is now UNREPRESENTABLE
    // rather than rejected at runtime — a total-function guarantee that is
    // strictly stronger than the guard this test covered. There is nothing
    // left to assert: the compiler enforces it.

    @Test func reservationExhaustionThrowsAtomically() throws {
        let kinds = [fullKind()]
        // Tiny pool: room for a handful of pages only. The +1 page is the
        // group's poison page, which is carved OUT of `capacityBytes`, so a
        // budget of N+1 pages is what buys N tenant pages.
        let pageBytes = 2 * 2 * 16 * 64 * 2
        let pool = try PagedKVPool(
            layerKinds: kinds,
            config: config(capacityBytes: pageBytes * 9, nominalMaxLen: 128))
        let key = PagedKVGroupKey(kvHeads: 2, headDim: 64)
        try pool.reserve([key: 6])
        #expect(throws: CBv2KVError.self) {
            try pool.reserve([key: 3])
        }
        // Failed reserve must leave no partial state.
        try pool.reserve([key: 2])
        pool.unreserve([key: 8])
        #expect(pool.bytesReserved == 0)
    }

    // MARK: - Sequence state: full attention

    @Test func fullSequenceRoundtrip() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        let row = try #require(state[0] as? PagedSequenceKV)

        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        for n in [5, 16, 12, 1] {
            let k = randomKV(heads: 2, tokens: n, dim: 64)
            let v = randomKV(heads: 2, tokens: n, dim: 64)
            mirrorK.append(k)
            mirrorV.append(v)
            let (viewK, viewV) = row.update(
                keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0))
            #expect(viewK.shape == [1, 2, row.retainedCount, 64])
            #expect(viewV.shape == viewK.shape)
        }
        #expect(row.absoluteOffset == 34)
        #expect(row.retainedCount == 34)
        // 34 tokens => 3 pages of 16.
        #expect(row.table.count == 3)
        #expect(backend.bytesInUse == 3 * backend.pool.group(row.groupKey).pageBytes)

        let expectedK = concatenated(mirrorK, axis: 1).expandedDimensions(axis: 0)
        let expectedV = concatenated(mirrorV, axis: 1).expandedDimensions(axis: 0)
        let snap = row.snapshot()
        #expect(snap.offset == 34)
        assertEqualArrays(snap.keys, expectedK)
        assertEqualArrays(snap.values, expectedV)

        backend.release(state)
        #expect(backend.bytesInUse == 0)
        #expect(backend.pool.bytesReserved == 0)
    }

    @Test func rollbackFreesTailPagesAndRewrites() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128)
        let row = try #require(state[0] as? PagedSequenceKV)

        let k0 = randomKV(heads: 2, tokens: 20, dim: 64)
        let v0 = randomKV(heads: 2, tokens: 20, dim: 64)
        row.write(keys: k0, values: v0)
        #expect(row.table.count == 2)

        row.rollback(6)
        #expect(row.absoluteOffset == 14)
        #expect(row.table.count == 1)

        // Rewrite different tail tokens; gather must reflect the new tail.
        let k1 = randomKV(heads: 2, tokens: 4, dim: 64)
        let v1 = randomKV(heads: 2, tokens: 4, dim: 64)
        row.write(keys: k1, values: v1)
        #expect(row.absoluteOffset == 18)

        let expectedK = concatenated(
            [k0[0..., 0 ..< 14, 0...], k1], axis: 1
        ).expandedDimensions(axis: 0)
        let (gotK, _) = row.attendableViews()
        assertEqualArrays(gotK, expectedK)
        backend.release(state)
    }

    // MARK: - Sequence state: sliding window ring

    @Test func windowedRingWrapKeepsRecentEnd() throws {
        let window = 32
        let kinds = [windowedKind(window)]
        let cfg = config(maxPrefillChunk: 16)
        let backend = try PagedKVBackend(layerKinds: kinds, config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)
        let ring = PagedKVPool.ringPageCount(window: window, config: cfg)
        // What THIS test needs of the ring, stated exactly: it gathers
        // `retainedCount == min(written, window)` after each write, and
        // `gatherRange` aborts once that span has been lapped. The page count
        // itself is pinned by `ringSpansExposedWindowPlusSpeculativeSpan` /
        // `ringIsTheSmallestSizeClearingTheFloor` — not re-derived here and
        // not hard-coded here, so a future sizing change fails the sizing
        // tests rather than three unrelated ones.
        #expect(PagedSequenceKV.maxWindowExposure(window: window) <= ring * cfg.pageSize)

        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        var written = 0
        for n in [16, 7, 16, 16, 3, 16, 16, 10] {
            let k = randomKV(heads: 2, tokens: n, dim: 64)
            let v = randomKV(heads: 2, tokens: n, dim: 64)
            mirrorK.append(k)
            mirrorV.append(v)
            row.write(keys: k, values: v)
            written += n
            #expect(row.absoluteOffset == written)
            #expect(row.table.count <= ring)

            let retained = row.retainedCount
            #expect(retained == min(written, window))
            let allK = concatenated(mirrorK, axis: 1)
            let allV = concatenated(mirrorV, axis: 1)
            let (gotK, gotV) = row.attendableViews()
            assertEqualArrays(
                gotK, allK[0..., (written - retained) ..< written, 0...]
                    .expandedDimensions(axis: 0))
            assertEqualArrays(
                gotV, allV[0..., (written - retained) ..< written, 0...]
                    .expandedDimensions(axis: 0))
        }
        // Ring must have wrapped (100 tokens through a 4-page/64-token ring).
        #expect(row.table.count == ring)
        backend.release(state)
        #expect(backend.bytesInUse == 0)
    }

    @Test func decodeAttendRangeClampsToWindow() throws {
        let kinds = [windowedKind(32)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 16))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)

        row.write(keys: randomKV(heads: 2, tokens: 10, dim: 64),
                  values: randomKV(heads: 2, tokens: 10, dim: 64))
        var range = row.decodeAttendRange
        #expect(range.start == 0 && range.length == 10)

        for _ in 0 ..< 6 {
            row.write(keys: randomKV(heads: 2, tokens: 16, dim: 64),
                      values: randomKV(heads: 2, tokens: 16, dim: 64))
        }
        // 106 tokens written; query position 105; window 32 => start 74.
        range = row.decodeAttendRange
        #expect(range.start == 74 && range.length == 32)
        backend.release(state)
    }

    // MARK: - Backend lifecycle

    @Test func sharedLayersGetNilStateAndValidation() throws {
        var shared = fullKind()
        shared.sharesKVWithLayer = 0
        let kinds = [fullKind(), shared]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 64)
        #expect(state.count == 2)
        #expect(state[0] != nil)
        #expect(state[1] == nil)
        backend.release(state)

        // Invalid head dim -> ineligible at build.
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(headDim: 96)], config: config())
        }
        // Bad GQA ratio -> ineligible at build.
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(kvHeads: 3, queryHeads: 4)], config: config())
        }
        // Shared layer pointing at another shared layer -> ineligible.
        var badShared = fullKind()
        badShared.sharesKVWithLayer = 1
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(), shared, badShared].map { $0 },
                config: config())
        }
    }

    @Test func admissionThrowsCapacityExhaustedAndRecovers() throws {
        let kinds = [fullKind()]
        let pageBytes = 2 * 2 * 16 * 64 * 2
        // Room for ~8 pages == 128 tokens.
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: config(capacityBytes: pageBytes * 8, nominalMaxLen: 128))

        let a = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 96)  // 6 pages
        #expect(throws: CBv2KVError.self) {
            _ = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 0, maxLength: 96)
        }
        backend.release(a)
        let b = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 96)
        backend.release(b)
    }

    @Test func adoptPrefixRoundtrip() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())

        // Donor sequence.
        let donor = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128)
        let donorRow = try #require(donor[0] as? PagedSequenceKV)
        let k = randomKV(heads: 2, tokens: 40, dim: 64)
        let v = randomKV(heads: 2, tokens: 40, dim: 64)
        donorRow.write(keys: k, values: v)
        let snap = donorRow.snapshot()
        // Materialize the snapshot before the donor pages are recycled —
        // gathered views reference the live slabs.
        eval(snap.keys, snap.values)
        backend.release(donor)

        let adopted = try backend.makeSequenceState(
            adopting: [(keys: snap.keys, values: snap.values, offset: snap.offset)],
            layerKinds: kinds, maxLength: 128)
        let row = try #require(adopted[0] as? PagedSequenceKV)
        #expect(row.absoluteOffset == 40)
        let (gotK, gotV) = row.attendableViews()
        assertEqualArrays(gotK, k.expandedDimensions(axis: 0))
        assertEqualArrays(gotV, v.expandedDimensions(axis: 0))
        backend.release(adopted)
    }

    @Test func fastForwardPlacesWindowedRecompute() throws {
        let kinds = [windowedKind(32)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 16))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)

        row.fastForward(to: 100)
        #expect(row.absoluteOffset == 100)
        #expect(row.retainedCount == 0)

        row.write(keys: randomKV(heads: 2, tokens: 16, dim: 64),
                  values: randomKV(heads: 2, tokens: 16, dim: 64))
        #expect(row.absoluteOffset == 116)
        // Only replayed tokens are attendable.
        #expect(row.retainedCount == 16)
        let range = row.decodeAttendRange
        #expect(range.start == 100 && range.length == 16)
        backend.release(state)
    }

    @Test func updateHonorsMaxLengthReservation() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 32)
        let row = try #require(state[0] as? PagedSequenceKV)
        row.write(keys: randomKV(heads: 2, tokens: 32, dim: 64),
                  values: randomKV(heads: 2, tokens: 32, dim: 64))
        #expect(row.table.count == 2)
        backend.release(state)
    }

    // MARK: - Sink gating

    /// A layer whose kind declares `hasSinks == false` must IGNORE a
    /// (mistakenly) passed sinks array — the same gate CBv2AttentionV1
    /// applies — on both the prefill and decode dispatch paths.
    /// (Deterministic position-coded inputs, NOT the global MLXRandom
    /// state: Swift Testing runs suites in parallel, and a concurrent test
    /// advancing the global RNG between runs would break the comparison.)
    @Test func sinksGatedByLayerKind() throws {
        let kinds = [fullKind()]
        #expect(!kinds[0].hasSinks)
        let garbageSinks = MLXArray(converting: [50.0, 50.0, 50.0, 50.0])

        func coded(_ shape: [Int], seed: Float) -> MLXArray {
            let count = shape.reduce(1, *)
            return MLXArray(0 ..< Int32(count)).asType(.float16)
                .reshaped(shape) * 0.001 + seed
        }

        func run(sinks: MLXArray?) throws -> (prefill: MLXArray, decode: MLXArray) {
            let backend = try PagedKVBackend(layerKinds: kinds, config: config())
            let cache = backend.makeLayerCaches()[0]
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 8, maxLength: 32)
            defer { backend.release(state) }
            cache.setRows([state[0]!])
            let prefill = cache.updateAndAttend(
                queries: coded([1, 4, 8, 64], seed: 0.3),
                keys: coded([1, 2, 8, 64], seed: 0.1),
                values: coded([1, 2, 8, 64], seed: 0.2),
                scale: 0.125, sinks: sinks)
            let decode = cache.updateAndAttend(
                queries: coded([1, 4, 1, 64], seed: 0.6),
                keys: coded([1, 2, 1, 64], seed: 0.4),
                values: coded([1, 2, 1, 64], seed: 0.5),
                scale: 0.125, sinks: sinks)
            eval(prefill, decode)
            return (prefill, decode)
        }

        let with = try run(sinks: garbageSinks)
        let without = try run(sinks: nil)
        assertEqualArrays(with.prefill, without.prefill)
        assertEqualArrays(with.decode, without.decode)
    }

    // MARK: - Device block-table cache identity

    /// Regression (cross-request page-table reuse): the device-table cache
    /// used to fingerprint rows by (ObjectIdentifier, tableVersion). A heap
    /// address can be recycled after a finished row deallocates, and — via
    /// the LIFO free list — the replacement row gets the SAME page ids and
    /// the SAME tableVersion trajectory, so a stale fingerprint hit would
    /// make the kernel attend the finished request's cached tables. The
    /// fingerprint is now a pool-issued monotonic serial: never reused, so
    /// two identical-geometry rows can NEVER collide, and any row-identity
    /// change forces a rebuild.
    @Test func deviceTablesRebuildOnRowIdentityReuse() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let cache = backend.makeLayerCaches()[0]

        func makeWrittenRow() throws -> [CBv2SequenceKV?] {
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 8, maxLength: 32)
            let row = try #require(state[0] as? PagedSequenceKV)
            row.write(
                keys: randomKV(heads: 2, tokens: 8, dim: 64),
                values: randomKV(heads: 2, tokens: 8, dim: 64))
            return state
        }

        let stateA = try makeWrittenRow()
        let rowA = try #require(stateA[0] as? PagedSequenceKV)
        cache.setRows([rowA])
        _ = cache.deviceTables(rows: [rowA])
        #expect(cache.tablesRebuildCount == 1)
        _ = cache.deviceTables(rows: [rowA])
        #expect(cache.tablesRebuildCount == 1, "unchanged table must hit the cache")
        let tableA = rowA.table
        let versionA = rowA.tableVersion

        // Release A, then allocate B with identical geometry: the LIFO free
        // list hands B the same physical page ids and B's tableVersion
        // trajectory matches A's — only the pool serial tells them apart.
        backend.release(stateA)
        let stateB = try makeWrittenRow()
        let rowB = try #require(stateB[0] as? PagedSequenceKV)
        #expect(rowB.table == tableA, "LIFO free list reuses the same page ids")
        #expect(rowB.tableVersion == versionA, "same allocation trajectory")
        #expect(rowB.serial != rowA.serial, "pool serials are never reused")

        cache.setRows([rowB])
        _ = cache.deviceTables(rows: [rowB])
        #expect(
            cache.tablesRebuildCount == 2,
            "a row-identity change must rebuild the device tables even when page ids and tableVersion match"
        )
        backend.release(stateB)
    }

    // MARK: - Admission-time page reservation (Codex P2)

    /// Reserving worst-case pages UP FRONT means several same-step admissions
    /// cannot over-commit the pool: the excess reservation throws before any
    /// request is accepted, and `bytesReserved` never exceeds capacity.
    @Test func admissionReservationCannotOvercommitPool() throws {
        let kinds = [fullKind()]  // one group, kv2 x d64
        // Pool sized for exactly 2 requests of maxLength 128 (8 pages each)
        // plus the group's poison page, which comes out of `capacityBytes`.
        let perRequestPages = PagedKVPool.pageDemand(
            kind: kinds[0], maxLength: 128,
            config: config(nominalMaxLen: 128))
        let pageBytes = 2 * 2 * 16 * 64 * 2
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: config(
                capacityBytes: pageBytes * (perRequestPages * 2 + 1), nominalMaxLen: 128))
        #expect(backend.bytesReserved == 0)

        try backend.reserve(layerKinds: kinds, maxLength: 128)
        try backend.reserve(layerKinds: kinds, maxLength: 128)
        #expect(backend.bytesReserved <= backend.bytesCapacity)
        let reservedAtCap = backend.bytesReserved

        // The third same-step admission cannot fit — it must throw, not
        // over-commit the pool.
        #expect(throws: CBv2KVError.self) {
            try backend.reserve(layerKinds: kinds, maxLength: 128)
        }
        #expect(backend.bytesReserved == reservedAtCap, "failed reserve leaves no partial state")

        // A pre-reserved request materializes WITHOUT double-charging, and
        // release returns exactly what admission held.
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128, reserved: true)
        #expect(backend.bytesReserved == reservedAtCap, "materialization must not re-charge")
        backend.release(state)
        backend.unreserve(layerKinds: kinds, maxLength: 128)  // release the 2nd hold
        #expect(backend.bytesReserved == 0, "every hold balanced")
    }

    // MARK: - position offsets: device-cached, membership-gated (Codex P2)

    private func decodeQKV(queryHeads: Int, kvHeads: Int, dim: Int, tokens: Int)
        -> (q: MLXArray, k: MLXArray, v: MLXArray)
    {
        (
            MLXRandom.normal([1, queryHeads, tokens, dim], dtype: .float16),
            MLXRandom.normal([1, kvHeads, tokens, dim], dtype: .float16),
            MLXRandom.normal([1, kvHeads, tokens, dim], dtype: .float16)
        )
    }

    @Test func positionOffsetsAdvanceOnDeviceWithoutHostRebuild() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        let cache = backend.makeLayerCaches()[0]
        let scale = 1.0 / Float(kind.headDim).squareRoot()

        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer { backend.release(state) }
        cache.setRows([state[0]!])
        #expect(cache.positionOffsetsHostRebuilds == 1, "setRows rebuilds once")

        // Prefill chunk of 4 tokens: offset 0 -> 4 on device, no host rebuild.
        let pf = decodeQKV(queryHeads: kind.queryHeads, kvHeads: kind.kvHeads, dim: kind.headDim, tokens: 4)
        _ = cache.updateAndAttend(queries: pf.q, keys: pf.k, values: pf.v, scale: scale, sinks: nil)
        #expect(cache.positionOffsetsHostRebuilds == 1)
        #expect(cache.positionOffsets.item(Int32.self) == 4)

        // 20 decode steps: offsets advance on device, counter stays 1.
        for _ in 0 ..< 20 {
            let d = decodeQKV(
                queryHeads: kind.queryHeads, kvHeads: kind.kvHeads, dim: kind.headDim, tokens: 1)
            _ = cache.updateAndAttend(queries: d.q, keys: d.k, values: d.v, scale: scale, sinks: nil)
        }
        #expect(cache.positionOffsetsHostRebuilds == 1, "decode loop must not host-rebuild offsets")
        #expect(cache.positionOffsets.item(Int32.self) == 24, "4 prefill + 20 decode tokens")

        // A batch membership change rebuilds from host truth.
        let state2 = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer { backend.release(state2) }
        cache.setRows([state[0]!, state2[0]!])
        #expect(cache.positionOffsetsHostRebuilds == 2, "membership change rebuilds")
    }

    // MARK: - windowed adoption decode-table length (Codex P2)

    private func assertClose(
        _ got: MLXArray, _ want: MLXArray, rtol: Float = 1e-2, atol: Float = 2e-3,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        let ok = allClose(
            got.asType(.float32), want.asType(.float32), rtol: Double(rtol), atol: Double(atol)
        ).item(Bool.self)
        if !ok {
            let diff = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
            Issue.record("arrays differ, max abs err \(diff)", sourceLocation: sourceLocation)
        }
    }

    /// A prefix-adopted windowed row whose trailing replay covers LESS than
    /// the ring leaves `table.count < ringPages`. The decode kernel must
    /// divide logical pages by the RING length, not `table.count`, or it
    /// wraps at the wrong divisor and reads the wrong physical pages — decode
    /// then diverges from a fresh-prefill reference. Token-exact regression.
    @Test func windowedAdoptionUsesRingLengthForDecodeTable() throws {
        let window = 32
        let kind = windowedKind(window, headDim: 64, kvHeads: 2, queryHeads: 4)
        let dim = kind.headDim
        let heads = kind.kvHeads
        let scale = 1.0 / Float(dim).squareRoot()
        // pageSize 16, maxPrefillChunk 16. The exact ring length is pinned by
        // the ring-sizing tests; here all that matters is that the ring is
        // longer than the two pages the trailing replay fills, so
        // `table.count < ring` below is a real condition and not an accident.
        // The adoption offset is DERIVED from the ring for the same reason:
        // at a hard-coded offset a resize can silently land the replay on the
        // ring's last slot and grow the table to its full length, which makes
        // the test vacuous rather than red.
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: config(capacityBytes: 32 << 20, maxPrefillChunk: 16, nominalMaxLen: 4096))
        let ring = PagedKVPool.ringPageCount(window: window, config: backend.pool.config)
        #expect(ring > 2, "replay of 2 pages must leave the ring partially allocated")

        // Deterministic position-coded K/V so a mis-resolved page is visible:
        // every absolute position gets a distinct vector.
        func coded(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
            let n = positions.count
            var kflat = [Float](repeating: 0, count: heads * n * dim)
            var vflat = [Float](repeating: 0, count: heads * n * dim)
            var i = 0
            for h in 0 ..< heads {
                for p in positions {
                    for d in 0 ..< dim {
                        kflat[i] = Float(p % 97) * 0.01 + Float(d) * 0.001 + Float(h) * 0.1
                        vflat[i] = Float(p % 97) * 0.02 - Float(d) * 0.0005 + Float(h) * 0.05
                        i += 1
                    }
                }
            }
            return (
                MLXArray(kflat, [heads, n, dim]).asType(.float16),
                MLXArray(vflat, [heads, n, dim]).asType(.float16)
            )
        }

        // Adopt on a ring-slot-0 boundary and replay two pages, so the replay
        // fills slots 0,1 and leaves the rest of the ring untouched.
        let adopt = 3 * ring * backend.pool.config.pageSize
        let cache = backend.makeLayerCaches()[0]
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 4096)
        defer { backend.release(state) }
        let row = state[0] as! PagedSequenceKV
        row.fastForward(to: adopt)
        for chunkStart in stride(from: adopt, to: adopt + 32, by: 16) {
            let (ck, cv) = coded(chunkStart ..< chunkStart + 16)
            row.write(keys: ck, values: cv)
        }
        #expect(row.table.count == 2, "replay must leave the ring partially allocated")
        #expect(row.table.count < ring, "replay must leave the ring partially allocated")
        #expect(row.decodeTableLength == ring, "windowed decode must divide by the ring length")
        cache.setRows([row])

        // Decode at `adopt + 32`: window [adopt + 1, adopt + 32].
        let qPos = adopt + 32
        let q = MLXRandom.normal([1, kind.queryHeads, 1, dim], dtype: .float16)
        let (nk, nv) = coded(qPos ..< (qPos + 1))
        let out = cache.updateAndAttend(
            queries: q,
            keys: nk.expandedDimensions(axis: 0),
            values: nv.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)

        // Reference: attention over the true window.
        let (wk, wv) = coded((qPos - window + 1) ..< (qPos + 1))
        let reference = PagedAttentionReference.composedAttention(
            queries: q, keys: wk.expandedDimensions(axis: 0),
            values: wv.expandedDimensions(axis: 0), scale: scale)
        assertClose(out, reference)
    }

    // MARK: - WS-1.2: the ring shrink, exercised
    //
    // The ring is `max(window + span, maxPrefillChunk)` tokens instead of
    // `window - 1 + maxPrefillChunk + span`, and it is only correct because
    // the gather moved BEFORE the write on both write paths and because
    // `PagedKVPool.gather` publishes a fence back-edge. The tests here run
    // the paths rather than restating the arithmetic: an earlier attempt at
    // this exact size passed every arithmetic test and aborted the daemon on
    // ordinary windowed prefill.

    /// Position-coded K/V. Every element is `(131p + 17h + d) mod 2039`, an
    /// integer below 2048 and therefore EXACT in float16; 2039 is prime and
    /// 131 is invertible modulo it, so two distinct positions can never
    /// produce the same code for a given (head, channel). A gathered range
    /// therefore names exactly the absolute positions it came from, and a
    /// clobbered ring slot cannot pass as intact.
    private func positionCoded(
        _ positions: Range<Int>, heads: Int = 2, dim: Int = 64
    ) -> (MLXArray, MLXArray) {
        let n = positions.count
        var kflat = [Float](repeating: 0, count: heads * n * dim)
        var vflat = kflat
        var i = 0
        for h in 0 ..< heads {
            for p in positions {
                for d in 0 ..< dim {
                    let code = (131 * p + 17 * h + d) % 2039
                    kflat[i] = Float(code)
                    vflat[i] = Float((code * 7) % 2039)
                    i += 1
                }
            }
        }
        return (
            MLXArray(kflat, [heads, n, dim]).asType(.float16),
            MLXArray(vflat, [heads, n, dim]).asType(.float16)
        )
    }

    /// ORDINARY WINDOWED PREFILL AT A FULL CHUNK, on the shrunk ring.
    ///
    /// This is the exact scenario the reverted attempt died on: a windowed
    /// row prefilled in `maxPrefillChunk` chunks where `window - 1 + chunk`
    /// exceeds the whole ring. Two things must hold at once, and only running
    /// it shows either:
    ///
    ///  * the chunk-shaped round must still SEE `min(written, window)` of
    ///    history — checked against an independent mirror, position by
    ///    position, so a clobbered slot cannot pass;
    ///  * the chunk's EARLIEST query must still see its full window — checked
    ///    by asserting the assembled view `update` returns is
    ///    `min(writtenBefore, window - 1) + n` wide and that its leading
    ///    columns are the pre-write history, not the chunk's own tail.
    ///
    /// Run at the production geometry (window 1,024, chunk 512, pageSize 16)
    /// and at small shapes, because the failure is a rounding-sensitive
    /// overlap between the gathered history and the chunk's ring slots.
    @Test(arguments: [
        (window: 32, chunk: 32), (window: 64, chunk: 48), (window: 1024, chunk: 512),
    ])
    func windowedPrefillAtFullChunkSurvivesTheRing(_ shape: (window: Int, chunk: Int)) throws {
        let (window, chunk) = shape
        let kind = windowedKind(window)
        let cfg = config(capacityBytes: 64 << 20, maxPrefillChunk: chunk, nominalMaxLen: 8192)
        let ringTokens = PagedKVPool.ringPageCount(window: window, config: cfg) * cfg.pageSize
        // The condition that makes this test meaningful: a POST-write gather
        // of the same span would not fit. If this ever stops holding the ring
        // grew back and the test is no longer exercising the shrink.
        #expect(
            window - 1 + chunk > ringTokens || chunk == 1,
            "window \(window) chunk \(chunk): ring \(ringTokens) is not tight enough to test")

        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 4 * window + 2 * chunk)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)

        var written = 0
        while written + chunk <= 3 * window + chunk {
            let (k, v) = positionCoded(written ..< (written + chunk))
            let historyBefore = min(written, window - 1)
            let (viewK, viewV) = row.update(keys: k, values: v)
            written += chunk

            // Width: history + chunk, WIDER than retainedCount by design.
            #expect(
                viewK.dim(2) == historyBefore + chunk,
                "chunk view is \(viewK.dim(2)) columns, expected \(historyBefore + chunk)")
            #expect(row.retainedCount == min(written, window))

            // Contents: exactly positions [written - chunk - historyBefore,
            // written). The leading columns must be the PRE-write history —
            // if the chunk's tail had lapped them, these are the wrong
            // positions and the codes do not match.
            let first = written - chunk - historyBefore
            let (wantK, wantV) = positionCoded(first ..< written)
            assertEqualArrays(viewK, wantK.expandedDimensions(axis: 0))
            assertEqualArrays(viewV, wantV.expandedDimensions(axis: 0))

            // And the ring itself still holds the trailing window.
            let (ringK, _) = row.attendableViews()
            let (wantRingK, _) = positionCoded((written - row.retainedCount) ..< written)
            assertEqualArrays(ringK, wantRingK.expandedDimensions(axis: 0))
        }
    }

    /// The DIRECT-WRITER path at the new ring size: `write` then read, with
    /// no gather-before-write trick available, at a full `maxPrefillChunk`.
    ///
    /// This is what the kernel differential harness
    /// (`CBv2PagedKernelTests.Fixture.addRow`) and `PagedDecodeProfiler` do.
    /// Their chunk bound used to be `ringTokens - window + 1` — 17 tokens at
    /// gemma-4's geometry — because a direct write left `retainedCount` at
    /// `window - 1 + n` and `attendableViews` then asked the ring for a span
    /// it had lapped. The bound is now `maxPrefillChunk`, which the ring
    /// covers by construction (`ringRowBoundDominatesALongChunk`).
    @Test(arguments: [(window: 32, chunk: 32), (window: 1024, chunk: 512)])
    func directWriterUsesTheWholeChunkAtTheNewRing(_ shape: (window: Int, chunk: Int)) throws {
        let (window, chunk) = shape
        let kind = windowedKind(window)
        let cfg = config(capacityBytes: 64 << 20, maxPrefillChunk: chunk, nominalMaxLen: 8192)
        let ringTokens = PagedKVPool.ringPageCount(window: window, config: cfg) * cfg.pageSize
        // The bound the harness used to be held to, kept as live arithmetic:
        // at gemma-4's geometry it is 17 against a 512-token chunk.
        let oldBound = max(1, min(chunk, ringTokens - window + 1))
        #expect(chunk > oldBound, "fixture is wrong: the old bound was not the binding one")

        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 8 * chunk + 2 * window)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)

        var written = 0
        for _ in 0 ..< 6 {
            let (k, v) = positionCoded(written ..< (written + chunk))
            row.write(keys: k, values: v)
            written += chunk
            #expect(row.absoluteOffset == written)
            // The read a direct writer actually performs. Under the old
            // semantics this asked for `window - 1 + chunk` and aborted.
            let (gotK, gotV) = row.attendableViews()
            #expect(row.retainedCount == min(written, window))
            let (wantK, wantV) = positionCoded((written - row.retainedCount) ..< written)
            assertEqualArrays(gotK, wantK.expandedDimensions(axis: 0))
            assertEqualArrays(gotV, wantV.expandedDimensions(axis: 0))
            // `snapshot()` is the same read and must be serviceable in the
            // same phase — the prefix cache donates from here.
            #expect(row.snapshot().offset == written)
        }
    }

    /// `PagedKVPool.gather` must publish a fence BACK-edge, so an in-place
    /// write cannot overtake a lazy gather that has not materialised.
    ///
    /// Without it the gather and the following `bulkWrite` are graph
    /// SIBLINGS: both merely consume the group's `writeFence`, and MLX may
    /// run either first. At the old 1,552-token ring that was invisible,
    /// because a chunk's history (`window - 1`) and the chunk itself never
    /// shared a ring slot. At 1,040 they do, so the loser of the race is the
    /// chunk's earliest queries reading the chunk's own tail as their
    /// history: a wrong answer with no crash and no telemetry.
    ///
    /// Two assertions, because either alone is weak. The mechanism: the
    /// fence is a NEW node after a gather and keeps its value. The effect: a
    /// gather held unevaluated across an overlapping write still reads the
    /// pre-write bytes.
    @Test func gatherPublishesAFenceBackEdgeSoWritesCannotOvertakeIt() throws {
        let window = 64
        let chunk = 64
        let kind = windowedKind(window)
        let cfg = config(capacityBytes: 32 << 20, maxPrefillChunk: chunk, nominalMaxLen: 4096)
        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 1024)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let group = backend.pool.group(row.groupKey)

        let (k0, v0) = positionCoded(0 ..< 128)
        for start in stride(from: 0, to: 128, by: chunk) {
            row.write(
                keys: k0[0..., start ..< (start + chunk), 0...],
                values: v0[0..., start ..< (start + chunk), 0...])
        }

        // MECHANISM.
        let fenceBefore = group.writeFence
        eval(fenceBefore)
        let valueBefore = fenceBefore.item(Int32.self)
        let (histK, histV) = row.gatherRange(start: 128 - (window - 1), count: window - 1)
        #expect(group.writeFence !== fenceBefore, "gather published no back-edge")
        eval(group.writeFence)
        #expect(
            group.writeFence.item(Int32.self) == valueBefore,
            "the back-edge must not change the fence's value")

        // EFFECT. `histK`/`histV` are still unevaluated. Write a chunk whose
        // ring slots OVERLAP them — which is exactly what a prefill chunk
        // does now that the ring is `window + span` rather than
        // `window - 1 + chunk + span`.
        let ringTokens = PagedKVPool.ringPageCount(window: window, config: cfg) * cfg.pageSize
        #expect(
            (window - 1) + chunk > ringTokens,
            "fixture is wrong: history and chunk do not share ring slots")
        let (k1, v1) = positionCoded(128 ..< (128 + chunk))
        row.write(keys: k1, values: v1)

        // Only NOW force the gather. It must still hold the pre-write bytes.
        let (wantK, wantV) = positionCoded((128 - (window - 1)) ..< 128)
        assertEqualArrays(histK, wantK.expandedDimensions(axis: 0))
        assertEqualArrays(histV, wantV.expandedDimensions(axis: 0))
    }

    // MARK: - Ring sizing: bounded on BOTH sides, from BOTH bounds (T4)
    //
    // `windowedRingWrapKeepsRecentEnd` asserted only `table.count <= ring`.
    // That is an UNDER-provision bound: it fires if the ring is too small for
    // the pages actually allocated and is silent about everything else, so
    // growing the ring — or failing to shrink it — fails nothing. The tests
    // below bracket it, and NONE re-derives its expectation from
    // `ringPageCount`: they state the requirement in tokens and let the page
    // count fall out.
    //
    // THE REQUIREMENT, in tokens. A windowed ring aliases at
    // `ringPages * pageSize`: writing absolute position `p` overwrites
    // `p - ringPages * pageSize`. TWO INDEPENDENT things must survive that,
    // and the ring is the larger of them:
    //
    //   CACHE  maxWindowExposure(window) + maxSpeculativeSpan
    //     `retainedCount` is `min(written, maxWindowExposure(window))` —
    //     `min(written, window)` — because both write paths gather a chunk's
    //     history BEFORE writing the chunk. `attendableViews()` gathers
    //     exactly that span and `gatherRange` aborts the process if it has
    //     been lapped. `maxSpeculativeSpan` on top is the alias margin a
    //     round spends before it can be rolled back; under-reserve it and a
    //     rejected draft destroys an entry still inside a live window —
    //     wrong output, no crash, no telemetry.
    //
    //   ROW    maxPrefillChunk
    //     One `PagedSequenceKV.write` scatters a whole chunk in ONE dispatch.
    //     Longer than the ring and two of its own tokens land in the same
    //     physical slot with no ordering between them. This bound USED to be
    //     implied by the chunk term the cache bound carried; it is not
    //     implied any more, and `ringRowBoundDominatesALongChunk` below is
    //     what stops it being dropped.

    private static let ringShapes = [
        (window: 32, chunk: 16),
        (window: 16, chunk: 16),
        (window: 512, chunk: 256),
        // gemma-4-26b-qat-4bit sliding layers under the production chunk size.
        (window: 1024, chunk: 512),
        (window: 1024, chunk: 64),
        // Row-bound: the chunk outruns the window, so the cache bound alone
        // would size this ring at 9 pages for a 128-page write.
        (window: 128, chunk: 2048),
    ]

    /// Tokens the ring must hold — see the requirement above.
    private static func ringFloorTokens(window: Int, chunk: Int) -> Int {
        max(
            PagedSequenceKV.maxWindowExposure(window: window)
                + CBv2PagedSpeculation.maxSpeculativeSpan,
            chunk)
    }

    /// LOWER bound — the safety floor. Under-sizing is a process abort on
    /// prefill (`gatherRange`) or silent corruption of confirmed history
    /// (speculative rollback or a self-lapping chunk), depending on which
    /// term is short.
    ///
    /// The cache term is the same quantity `PagedSequenceKV
    /// .speculativeHeadroom` reads from the row side, so the two cannot be
    /// allowed to disagree.
    @Test(arguments: ringShapes)
    func ringSpansExposedWindowPlusSpeculativeSpan(_ shape: (window: Int, chunk: Int)) {
        let cfg = config(maxPrefillChunk: shape.chunk)
        let ring = PagedKVPool.ringPageCount(window: shape.window, config: cfg)
        let floor = Self.ringFloorTokens(window: shape.window, chunk: shape.chunk)
        #expect(
            ring * cfg.pageSize >= floor,
            """
            ring \(ring) pages (\(ring * cfg.pageSize) tokens) is below the floor \(floor) = \
            max(exposure \(PagedSequenceKV.maxWindowExposure(window: shape.window)) + \
            maxSpeculativeSpan \(CBv2PagedSpeculation.maxSpeculativeSpan), \
            maxPrefillChunk \(shape.chunk)). Short by \(floor - ring * cfg.pageSize) tokens: \
            prefill will abort in gatherRange, a rolled-back speculative write will alias a \
            live in-window entry, or one chunk will lap the ring inside a single dispatch.
            """)
    }

    /// UPPER bound — the ring is the SMALLEST page count clearing the floor,
    /// so `ring - 1` pages must NOT clear it.
    ///
    /// Without this, nothing distinguishes a correctly sized ring from one
    /// carrying arbitrary slack, and a shrink could "land" as a no-op with
    /// every test still green. The bound is stated as an inequality rather
    /// than a literal page count precisely so that it keeps holding through
    /// the sizing changes that are still coming.
    ///
    /// HISTORY, so the next reader does not re-litigate it. The ring used to
    /// be `ceil((window - 1 + chunk) / pageSize) + ceil(span / pageSize)` —
    /// 97 pages for gemma-4, 1,552 tokens for a 1,024-token window, and a
    /// measured 1.10x per-sequence KV regression against the contiguous
    /// backend at 10k context. An earlier attempt to shrink it to 65 pages
    /// was REVERTED because 65 holds the window but not `window - 1 + chunk`,
    /// and a 512-token chunk asked `gatherRange` for 1,535 tokens against
    /// 1,040: a daemon abort on ordinary prefill, reproduced from the row
    /// side. What made 65 correct was NOT this arithmetic — it was moving the
    /// gather BEFORE the write on both the layer path
    /// (`PagedLayerCache.prefillKV`) and the row path
    /// (`PagedSequenceKV.update`), and publishing a fence back-edge in
    /// `PagedKVPool.gather` so the chunk write cannot overtake the gather now
    /// that they share ring slots. `windowedPrefillAtFullChunkSurvivesTheRing`
    /// and the seam suite's `attendableSpanIsWhatARealRowActuallyExposes`
    /// are what hold that condition; this test only pins the size.
    @Test(arguments: ringShapes)
    func ringIsTheSmallestSizeClearingTheFloor(_ shape: (window: Int, chunk: Int)) {
        let cfg = config(maxPrefillChunk: shape.chunk)
        let ring = PagedKVPool.ringPageCount(window: shape.window, config: cfg)
        let floor = Self.ringFloorTokens(window: shape.window, chunk: shape.chunk)
        #expect(
            (ring - 1) * cfg.pageSize < floor,
            """
            ring \(ring) pages (\(ring * cfg.pageSize) tokens) is over-provisioned: \
            \(ring - 1) pages (\((ring - 1) * cfg.pageSize) tokens) already clear the floor \
            \(floor), so at least \(cfg.pageSize) tokens per windowed layer are slack. On \
            gemma-4 that is 25 of 30 layers.
            """)
    }

    /// The ROW bound is not a corollary of the cache bound and must be able
    /// to DOMINATE it. This is the test that fails if someone re-derives the
    /// ring from the window alone.
    ///
    /// It is deliberately arithmetic-only and separate from the sweep above:
    /// the sweep would still pass on a formula that took `max` of the wrong
    /// two things, because five of its six shapes are cache-bound.
    @Test func ringRowBoundDominatesALongChunk() {
        let span = CBv2PagedSpeculation.maxSpeculativeSpan
        for (window, chunk) in [(128, 2048), (32, 512), (16, 4096)] {
            let cfg = config(maxPrefillChunk: chunk)
            let cacheOnly = PagedSequenceKV.maxWindowExposure(window: window) + span
            #expect(chunk > cacheOnly, "fixture is wrong: \(window)/\(chunk) is not row-bound")
            let ringTokens =
                PagedKVPool.ringPageCount(window: window, config: cfg) * cfg.pageSize
            #expect(
                ringTokens >= chunk,
                """
                window \(window) chunk \(chunk): the ring holds \(ringTokens) tokens and must \
                hold one whole maxPrefillChunk write (\(chunk)). The cache bound alone gives \
                \(cacheOnly) tokens — \((cacheOnly + cfg.pageSize - 1) / cfg.pageSize) pages \
                for a \((chunk + cfg.pageSize - 1) / cfg.pageSize)-page write — so a ring \
                derived from the window alone lets a chunk lap itself inside one bulk-write \
                dispatch, with no ordering between the two tokens landing in the same slot.
                """)
            // ...and the pool must actually BUILD at that geometry, which is
            // the half pure arithmetic cannot show: `checkedRingPageCount`
            // guards both bounds and would refuse it if either were short.
            #expect(throws: Never.self) {
                _ = try PagedKVPool(
                    layerKinds: [self.windowedKind(window)],
                    config: self.config(capacityBytes: 64 << 20, maxPrefillChunk: chunk))
            }
        }
    }

    // MARK: - Slab canary: pad targets must never name a live tenant page (T4)
    //
    // Two sites pad a device-visible index array up to the >= 8 entries the
    // generated kernel signatures need in order to keep their `device` address
    // space:
    //
    //   PagedKVPool.writeTokens   pads `slots` by repeating `slots[last]` —
    //                             a live physical slot of the writing row.
    //   PagedLayerCache.deviceTables
    //                             pads the block-table columns with literal 0
    //                             — page id 0, which today is an ordinary
    //                             allocatable page owned by whichever row
    //                             allocated first.
    //
    // Neither pad entry is read today: `bulkWrite` dispatches a grid sized by
    // the TRUE token count, and the decode kernel bounds its page walk by the
    // attended range. So this is not a live corruption — it is a failure MODE.
    // If either bound ever slips by one, the write lands in a live tenant's KV
    // and the only symptom is another request's output going quietly wrong.
    //
    // Track P (WS-1.1, this wave) reserves page 0 as a permanently-zeroed,
    // never-allocatable POISON page and repoints both pads at it. Page 0 is
    // the deliberate choice: `MLXArray.zeros`, `[Int32](repeating: 0, ...)`
    // and every default-initialised int32 buffer produce 0, so making 0
    // unallocatable converts the whole "forgot to pad / uninitialised table
    // entry" class into a read of zeros.

    /// Track P's poison-page surface is declared at the top of this file (it
    /// cannot be nested in a type). See `CBv2PoisonPageProviding` for why the
    /// binding is a protocol with sentinel defaults rather than a direct call.

    /// The poison page is never reserved, never freed and never allocatable;
    /// a live tenant's pages, its neighbours, and every pad entry must all be
    /// distinguishable from it. Sentinels are out-of-band so they cannot be
    /// mistaken for a real answer.
    @Test func poisonPageIsReservedAndNeverAllocatable() throws {
        let kinds = [fullKind()]
        let pool = try PagedKVPool(layerKinds: kinds, config: config())
        let key = PagedKVGroupKey(kvHeads: 2, headDim: 64)
        let pageCount = pool.group(key).pageCount

        let poison = pool.poisonPage(group: key)
        try #require(
            poison != poisonPageAbsent,
            """
            PagedKVPool.poisonPage(group:) is absent — track P's WS-1.1 reserved \
            poison page has not landed. Both pad sites still name a live page.
            """)
        #expect(
            poison == 0,
            "page 0 must be the poison page: it is the value every zeroed index buffer holds")
        #expect(!pool.isAllocatablePage(poison, group: key))
        #expect(
            pool.usablePageCount(group: key) == pageCount - 1,
            "the poison page must be excluded from usable capacity, not merely skipped")

        // Drain the group: no allocation may ever hand back the poison page,
        // and the drain must stop exactly one page short of the slab.
        var allocated: [Int32] = []
        while allocated.count < pool.usablePageCount(group: key) {
            let page = pool.allocatePage(group: key)
            #expect(page != poison, "allocatePage returned the poison page")
            #expect(pool.isAllocatablePage(page, group: key))
            allocated.append(page)
        }
        #expect(allocated.count == pageCount - 1)
        #expect(Set(allocated).count == allocated.count, "duplicate page handed out")
        pool.freePages(group: key, pages: allocated)
    }

    /// The poison page is all zeros and STAYS all zeros across a bulk write
    /// short enough to trigger the `slots` pad (`n < 8`).
    ///
    /// This is the direct oracle for `PagedKVPool.writeTokens`, whose `slots`
    /// array never escapes the function: a pad that duplicates a live slot
    /// leaves the poison page clean too, so the poison page being clean is
    /// only evidence once the pad targets it — hence the paired
    /// `poisonPageIsReservedAndNeverAllocatable` above. Together they say: the
    /// pad names the poison page, and nothing ever writes the poison page.
    ///
    /// EXPECTED RED until track P lands WS-1.1.
    @Test func poisonPageStaysZeroedAcrossAShortBulkWrite() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let pool = backend.pool
        let key = PagedKVGroupKey(kvHeads: 2, headDim: 64)
        let poison = pool.poisonPage(group: key)
        try #require(
            poison != poisonPageAbsent,
            """
            PagedKVPool.poisonPage(group:) is absent — track P's WS-1.1 reserved \
            poison page has not landed.
            """)

        func poisonPageContents() -> (MLXArray, MLXArray) {
            let (k, v) = pool.gather(
                group: key, pages: [poison], firstSlot: 0, count: pool.config.pageSize)
            eval(k, v)
            return (k, v)
        }
        let zeros = MLXArray.zeros([1, 2, pool.config.pageSize, 64], dtype: pool.config.dtype)
        let (k0, v0) = poisonPageContents()
        assertEqualArrays(k0, zeros)
        assertEqualArrays(v0, zeros)

        // n = 3 < 8 ⇒ writeTokens pads the slots array five times over.
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 64)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        row.write(
            keys: randomKV(heads: 2, tokens: 3, dim: 64) + 7.0,
            values: randomKV(heads: 2, tokens: 3, dim: 64) - 7.0)

        let (k1, v1) = poisonPageContents()
        assertEqualArrays(k1, zeros)
        assertEqualArrays(v1, zeros)
    }

    /// Behavioural canary for the SAME two pad sites, independent of whether
    /// the poison page exists: a live tenant's KV on both sides of the writing
    /// row must be bitwise unchanged after a short bulk write and after a
    /// decode dispatch whose block table is mostly pad columns.
    ///
    /// Passes today (the pads are provably unread) and is the assertion that
    /// fires if either kernel bound ever slips.
    @Test func shortBulkWriteAndPaddedDecodeLeaveNeighbourPagesIntact() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let cache = backend.makeLayerCaches()[0]
        let canaryTokens = 32

        /// Position-coded so a single overwritten slot is visible, and nowhere
        /// near the zeros an untouched page holds.
        func canary(_ tag: Float) -> (MLXArray, MLXArray) {
            let base = MLXArray(0 ..< Int32(2 * canaryTokens * 64)).asType(.float16)
                .reshaped([2, canaryTokens, 64]) * 0.0007
            return (base + tag, base - tag)
        }

        struct Tenant {
            let state: [CBv2SequenceKV?]
            let row: PagedSequenceKV
            let keys: MLXArray
            let values: MLXArray
        }

        func tenant(_ tag: Float) throws -> Tenant {
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 0, maxLength: 128)
            let row = try #require(state[0] as? PagedSequenceKV)
            let (k, v) = canary(tag)
            row.write(keys: k, values: v)
            let (gotK, gotV) = row.gatherRange(start: 0, count: canaryTokens)
            eval(gotK, gotV)  // materialize BEFORE anything else writes the slab
            return Tenant(state: state, row: row, keys: gotK, values: gotV)
        }

        func makeRow() throws -> (state: [CBv2SequenceKV?], row: PagedSequenceKV) {
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 0, maxLength: 128)
            return (state, try #require(state[0] as? PagedSequenceKV))
        }

        // Pages come off a LIFO free list low-first, so allocating in this
        // order brackets both writers between two canaries: an off-by-one slip
        // in either direction lands on live KV.
        let below = try tenant(3.0)

        // (a) `PagedKVPool.writeTokens` pad — every n below the 8-entry
        //     threshold, so the pad repeats between 1 and 7 times.
        let padWriter = try makeRow()
        for n in 1 ... 7 {
            padWriter.row.write(
                keys: randomKV(heads: 2, tokens: n, dim: 64),
                values: randomKV(heads: 2, tokens: n, dim: 64))
        }

        // (b) `PagedLayerCache.deviceTables` pad — this row holds ONE page
        //     against the 8-column minimum, so columns 1...7 of its block-table
        //     row are pad entries the decode kernel must never dereference.
        let shortTable = try makeRow()
        shortTable.row.write(
            keys: randomKV(heads: 2, tokens: 1, dim: 64),
            values: randomKV(heads: 2, tokens: 1, dim: 64))

        let above = try tenant(-3.0)
        #expect(
            Set(below.row.table).isDisjoint(with: Set(above.row.table)),
            "tenants must not share pages")
        #expect(shortTable.row.table.count == 1, "the padded-decode row must hold one page")

        cache.setRows([shortTable.row, padWriter.row])
        let stepK = randomKV(heads: 2, tokens: 1, dim: 64).expandedDimensions(axis: 0)
        let stepV = randomKV(heads: 2, tokens: 1, dim: 64).expandedDimensions(axis: 0)
        _ = cache.updateAndAttend(
            queries: MLXRandom.normal([2, 4, 1, 64], dtype: .float16),
            keys: concatenated([stepK, stepK], axis: 0),
            values: concatenated([stepV, stepV], axis: 0),
            scale: 0.125, sinks: nil)

        // Compare the canary RANGE, not the whole row: neither tenant is in
        // the dispatch, so their first `canaryTokens` positions must be
        // byte-identical to what they wrote.
        for (label, tenant) in [("below", below), ("above", above)] {
            let (gotK, gotV) = tenant.row.gatherRange(start: 0, count: canaryTokens)
            #expect(
                arrayEqual(gotK, tenant.keys).item(Bool.self),
                "\(label) tenant's keys were overwritten by a padded dispatch")
            #expect(
                arrayEqual(gotV, tenant.values).item(Bool.self),
                "\(label) tenant's values were overwritten by a padded dispatch")
        }

        backend.release(below.state)
        backend.release(above.state)
        backend.release(padWriter.state)
        backend.release(shortTable.state)
    }

    // MARK: - Prefix adoption vs a COLD row (T3)

    /// `adoptPrefixRoundtrip` above proves an adopted row reads back what was
    /// donated. It cannot see a residency/plan mismatch, because it never
    /// decodes: the signature of that bug is not a crash but a silently
    /// TRUNCATED window for the first W decoded tokens. An adopted row's
    /// windowed layers are `fastForward`ed to the replay start C, which also
    /// sets their `baseOffset` — so if C is off, or the trailing replay is
    /// short, every query whose window reaches behind C attends fewer keys
    /// than it should and the output is merely slightly wrong.
    ///
    /// The only oracle for that is a COLD row decoded from the same prompt.
    /// Greedy decoding is the amplifier: a truncated window perturbs the
    /// logits, the argmax flips once, and the trajectories separate for good.
    @Test func adoptedRowGreedyDecodeMatchesColdRow() throws {
        // [full, sliding(16)] — full FIRST, so the layout is paged-eligible
        // (an owning full layer after a windowed one is `pagedHybridRequires
        // DualCursor`). headDim 64: the paged kernel supports {64,128,256,512}.
        let model = TinyTestModel.make(seed: 0xADD_0FF, headDim: 64)
        let kinds = model.layerKinds
        let window = model.config.windowSize
        #expect(kinds[0].attention == .full)
        #expect(kinds[1].attention == .slidingWindow(window))

        let prompt = makePromptTokens(length: 49, seed: 0xB0A7)
        let chunkSize = 16
        // Three full window turnovers, so a window that is short by even one
        // key has ~48 chances to flip an argmax.
        let decodeSteps = 3 * window

        func makeBackend() throws -> PagedKVBackend {
            try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 32 << 20, maxPrefillChunk: chunkSize,
                    nominalMaxSequenceLength: 512))
        }

        /// Prefill `prompt[from ..< count-1]` in chunks, then greedy-decode.
        func run(
            backend: PagedKVBackend, state: [CBv2SequenceKV?], from: Int, steps: Int
        ) -> [Int] {
            let caches: [CBv2AttendingLayerCache] = backend.makeLayerCaches()
            for (i, kind) in kinds.enumerated() where kind.sharesKVWithLayer == nil {
                caches[i].setRows([state[i]!])
            }
            var index = from
            while index < prompt.count - 1 {
                let slice = Array(prompt[index ..< min(index + chunkSize, prompt.count - 1)])
                _ = model.forward(
                    tokens: MLXArray(slice.map(Int32.init)).reshaped(1, slice.count),
                    caches: caches)
                index += slice.count
            }
            var current = prompt.last!
            var generated: [Int] = []
            for _ in 0 ..< steps {
                let logits = model.forward(
                    tokens: MLXArray([Int32(current)]).reshaped(1, 1), caches: caches)
                current = Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
                generated.append(current)
            }
            return generated
        }

        // --- Cold arm: no adoption at all.
        let coldBackend = try makeBackend()
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: 256)
        let cold = run(backend: coldBackend, state: coldState, from: 0, steps: decodeSteps)
        coldBackend.release(coldState)

        // --- Adopted arm.
        // The planner's conservative replay bound is windowedLayers x window
        // = 16, so a 32-token match restores the full layer to C = 16 and
        // replays [16, 32). Everything from 32 on is ordinary prefill.
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        #expect(capability.strategy == .tailReplay)
        let plan = try #require(
            capability.plan(matchedBoundary: 32, maximumSequenceLength: 256))
        #expect(plan.replayStart == 16)
        #expect(plan.restoredFullTokens == 16)

        let adoptBackend = try makeBackend()
        // Donor: a cold prefill of prompt[0 ..< C). The full layer is the
        // FIRST layer, so its K/V over [0, C) is identical whether the donor
        // saw C tokens or the whole prompt.
        let donorState = try adoptBackend.makeSequenceState(
            layerKinds: kinds, promptLength: plan.replayStart, maxLength: 256)
        let donorCaches: [CBv2AttendingLayerCache] = adoptBackend.makeLayerCaches()
        for (i, kind) in kinds.enumerated() where kind.sharesKVWithLayer == nil {
            donorCaches[i].setRows([donorState[i]!])
        }
        _ = model.forward(
            tokens: MLXArray(prompt[0 ..< plan.replayStart].map(Int32.init))
                .reshaped(1, plan.replayStart),
            caches: donorCaches)
        let donated = donorState[0]!.snapshot()
        // Paged snapshots are lazy views over the shared slabs — materialize
        // before the donor's pages are recycled.
        eval(donated.keys, donated.values)
        #expect(donated.offset == plan.replayStart)
        adoptBackend.release(donorState)

        let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?] = [
            (keys: donated.keys, values: donated.values, offset: donated.offset), nil,
        ]
        let adoptedState = try adoptBackend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: 256)
        let windowedRow = try #require(adoptedState[1] as? PagedSequenceKV)
        #expect(adoptedState[0]?.absoluteOffset == plan.replayStart)
        #expect(
            windowedRow.absoluteOffset == plan.replayStart && windowedRow.retainedCount == 0,
            "the windowed layer must be fast-forwarded to C with nothing retained")
        let adopted = run(
            backend: adoptBackend, state: adoptedState, from: plan.replayStart,
            steps: decodeSteps)
        adoptBackend.release(adoptedState)

        let matched = zip(adopted, cold).prefix(while: { $0.0 == $0.1 }).count
        #expect(
            adopted == cold,
            """
            adopted row diverged from a cold row decoded from the same prompt \
            (\(matched) of \(decodeSteps) tokens matched before divergence) — \
            residency/plan mismatch
            """)
    }
}

// MARK: - Speculative (MTP) transactions — WS-3.2 / WS-3.3

/// Row-side speculative transaction: eligibility DERIVED from ring headroom
/// (WS-3.3) plus the bookkeeping-only round (WS-3.2) — speculative base,
/// tightened rollback bound, deferred page frees, restored `retainedCount`
/// input.
///
/// The claim under test is that a paged windowed ring aliases at
/// `ringPages * pageSize`, NOT at `window`, so a round overwrites only
/// positions that were already evicted and no data staging is needed. These
/// tests pin it from the outside: after a fully rolled-back round the ROW
/// (offset, retainedCount, page table, gathered bytes) and the POOL (free
/// list, refcounts, deferred queue) are indistinguishable from before it.
@Suite("CBv2PagedBackend: speculative rounds")
struct CBv2PagedSpeculativeRowTests {

    private let heads = 2
    private let dim = 64

    private func fullKind() -> CBv2LayerKind {
        CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    private func windowedKind(_ window: Int) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window), headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    /// `maxPrefillChunk` is deliberately SMALL relative to the window.
    ///
    /// The ring must cover `max(window + maxSpeculativeSpan, chunk)`, so a
    /// chunk under the window keeps these suites cache-bound — window 32 /
    /// chunk 8 needs 40 tokens and rings at 48. A larger chunk would move the
    /// ring for a reason that has nothing to do with the speculative
    /// transaction. The ring's own adequacy is asserted by
    /// `ringCoversTheExposedWindowPlusOneRound`.
    private func config(maxPrefillChunk: Int = 8) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: 8 << 20, maxPrefillChunk: maxPrefillChunk,
            nominalMaxSequenceLength: 1024)
    }

    /// Position-coded KV. Every element is `(131p + 17h + d) mod 2039`, an
    /// integer below 2048 and therefore EXACT in float16; 2039 is prime and
    /// 131 is invertible modulo it, so two distinct positions can never
    /// produce the same code for a given (head, channel). A gathered range
    /// therefore names exactly the absolute positions it came from, and a
    /// clobbered ring slot cannot pass as intact.
    private func coded(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
        let n = positions.count
        var kflat = [Float](repeating: 0, count: heads * n * dim)
        var vflat = kflat
        var i = 0
        for h in 0 ..< heads {
            for p in positions {
                for d in 0 ..< dim {
                    let code = (131 * p + 17 * h + d) % 2039
                    kflat[i] = Float(code)
                    vflat[i] = Float((code * 7) % 2039)
                    i += 1
                }
            }
        }
        return (
            MLXArray(kflat, [heads, n, dim]).asType(.float16),
            MLXArray(vflat, [heads, n, dim]).asType(.float16)
        )
    }

    private func write(_ row: PagedSequenceKV, _ positions: Range<Int>) {
        let (k, v) = coded(positions)
        row.write(keys: k, values: v)
    }

    /// Prefill `positions` as back-to-back `maxPrefillChunk` writes, so the
    /// row reaches the round through the same shape a real prefill would.
    private func fill(_ row: PagedSequenceKV, _ positions: Range<Int>, chunk: Int) {
        var p = positions.lowerBound
        while p < positions.upperBound {
            let n = min(chunk, positions.upperBound - p)
            write(row, p ..< (p + n))
            p += n
        }
    }

    /// Force the lazy gather to host so it survives later slab mutation —
    /// an unevaluated view would silently re-read the ring after the round.
    private func hostCopy(_ a: MLXArray) -> [Float] {
        a.asType(.float32).asArray(Float.self)
    }

    /// Everything a caller can observe about a row.
    private struct RowFingerprint: Equatable {
        var absoluteOffset: Int
        var retainedCount: Int
        var byteCount: Int
        var table: [Int32]
        var attendStart: Int
        var attendLength: Int
        var keys: [Float]
        var values: [Float]
    }

    private func fingerprint(_ row: PagedSequenceKV) -> RowFingerprint {
        let (k, v) = row.attendableViews()
        let range = row.decodeAttendRange
        return RowFingerprint(
            absoluteOffset: row.absoluteOffset, retainedCount: row.retainedCount,
            byteCount: row.byteCount, table: row.table,
            attendStart: range.start, attendLength: range.length,
            keys: hostCopy(k), values: hostCopy(v))
    }

    /// Everything a caller can observe about a group's page accounting.
    private struct PoolFingerprint: Equatable {
        var freeList: [Int32]
        var refCounts: [Int]
        var pagesInUse: Int
        var deferredFrees: [Int32]
    }

    private func fingerprint(_ pool: PagedKVPool, _ key: PagedKVGroupKey) -> PoolFingerprint {
        let g = pool.group(key)
        return PoolFingerprint(
            freeList: g.freeList, refCounts: g.refCounts, pagesInUse: g.pagesInUse,
            deferredFrees: g.deferredFrees)
    }

    // MARK: - WS-3.3 eligibility

    @Test func eligibilityIsDerivedFromRingHeadroom() throws {
        let window = 32
        let cfg = config()
        let kinds = [fullKind(), windowedKind(window)]
        let backend = try PagedKVBackend(layerKinds: kinds, config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        defer { backend.release(state) }
        let full = try #require(state[0] as? PagedSequenceKV)
        let windowed = try #require(state[1] as? PagedSequenceKV)

        // Asserted as a RELATION, not a magic number: `ringPageCount` is
        // track P's and is being resized (WS-1.2/3.1), so pinning "7 pages"
        // here would just rot into a false failure.
        let ring = PagedKVPool.ringPageCount(window: window, config: cfg)
        #expect(windowed.speculativeHeadroom == ring * cfg.pageSize - window)
        #expect(windowed.speculativeHeadroom >= CBv2PagedSpeculation.maxSpeculativeSpan)
        #expect(windowed.speculativeHeadroom < Int.max, "a windowed ring is bounded")
        #expect(full.speculativeHeadroom == Int.max, "full rows have no alias distance")

        // The gate is the DERIVATION, not the attention kind: the windowed
        // row used to be refused outright by `windowSize == nil`.
        for row in [full, windowed] {
            #expect(
                row.supportsSpeculativeWrites
                    == (row.speculativeHeadroom >= CBv2PagedSpeculation.maxSpeculativeSpan))
            #expect(row.supportsSpeculativeWrites)
        }
        #expect(
            EngineLoopV2.mtpStorageEligible(state),
            "a gemma-4-shaped hybrid bank must now reach the MTP storage gate")
    }

    /// Lower bound on the ring, from the row side.
    ///
    /// A windowed row can be asked for `retainedCount` positions at once, and
    /// that is `min(written, maxWindowExposure(window))`. On top of that a
    /// speculative round writes `maxSpeculativeSpan` past the frontier, and
    /// writing position `p` destroys whatever held `p - ringTokens`. So the
    /// ring must cover both at once or a windowed row gathers positions the
    /// ring already evicted — which is a `precondition` in `gatherRange`,
    /// i.e. a daemon abort on an ordinary prefill, no speculation required.
    ///
    /// Owner of the formula is `PagedKVPool.ringPageCount`. This test asserts
    /// only the property `speculativeHeadroom` depends on, which is why it
    /// does not mention `maxPrefillChunk`: the ROW bound is pinned separately
    /// by `ringRowBoundDominatesALongChunk`, and conflating the two is how a
    /// resize drops one of them.
    @Test func ringCoversTheExposedWindowPlusOneRound() {
        for (window, chunk) in [(32, 8), (512, 256), (1024, 64), (1024, 512)] {
            let cfg = PagedKVPoolConfig(
                capacityBytes: 8 << 20, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: 1024)
            let ringTokens = PagedKVPool.ringPageCount(window: window, config: cfg) * cfg.pageSize
            let exposure = PagedSequenceKV.maxWindowExposure(window: window)
            let widest = exposure + CBv2PagedSpeculation.maxSpeculativeSpan
            let why =
                "window \(window) chunk \(chunk): ring \(ringTokens) tokens cannot hold "
                + "retainedCount \(exposure) plus a "
                + "\(CBv2PagedSpeculation.maxSpeculativeSpan)-token round"
            #expect(widest <= ringTokens, Comment(rawValue: why))
        }
    }

    // MARK: - WS-3.2 round is pure bookkeeping

    @Test func windowedRoundFullyRolledBackRestoresRowAndPool() throws {
        let window = 32
        let cfg = config()
        let kind = windowedKind(window)
        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)

        // Fill several times past the ring so wrap-around aliasing is LIVE,
        // in `maxPrefillChunk` chunks — the shape a real prefill arrives in.
        let chunk = cfg.maxPrefillChunk
        fill(row, 0 ..< 104, chunk: chunk)
        #expect(row.absoluteOffset == 104)
        #expect(row.retainedCount == window, "exposure is the window, whatever the last write")
        let ringTokens = PagedKVPool.ringPageCount(window: window, config: cfg) * cfg.pageSize
        #expect(row.absoluteOffset > ringTokens, "the ring must have wrapped")

        let rowBefore = fingerprint(row)
        let poolBefore = fingerprint(backend.pool, row.groupKey)
        let versionBefore = row.tableVersion

        row.beginSpeculativeWrite()
        #expect(row.speculativeBase == 104)
        write(row, 104 ..< 112)  // 1 target + 7 drafts == maxSpeculativeSpan
        #expect(row.absoluteOffset == 112)
        row.rollback(8)
        row.commitSpeculativeWrite()
        #expect(row.speculativeBase == nil)

        // Row: byte-identical, including the page table. The round's writes
        // landed on slots whose prior tenants were `ringTokens` positions
        // behind the frontier — already evicted, outside every attendable
        // range. That is exactly the `ringTokens - window >= span` headroom
        // the ring reserves, and `roundBeyondTheRingHeadroomDestroysHistory`
        // below is the control showing it is what makes this work.
        #expect(fingerprint(row) == rowBefore)
        #expect(row.tableVersion == versionBefore, "a windowed ring allocates nothing in a round")
        // Pool: a windowed ring frees no pages, so nothing may be queued.
        #expect(fingerprint(backend.pool, row.groupKey) == poolBefore)
    }

    /// Control for the test above: the restoration is a property of the
    /// ring's RESERVED HEADROOM, not of rollback being free.
    ///
    /// Without a control, `windowedRoundFullyRolledBackRestoresRowAndPool`
    /// could pass on a row that simply never overwrites anything, and the
    /// reader would learn nothing about why `ringPageCount` carries
    /// `maxSpeculativeSpan`. So: write PAST the headroom the ring reserves
    /// and the same rollback does NOT restore the row — the round's writes
    /// reached into the still-attendable window and destroyed confirmed
    /// history.
    ///
    /// Unarmed on purpose. `guardSpeculativeSpan` traps this inside a
    /// transaction, which is the whole point of the guard; running it bare
    /// is what lets the test observe the corruption the guard prevents.
    ///
    /// THE DAMAGE IS OBSERVED IN TWO PLACES because a gather can no longer
    /// see it. `PagedSequenceKV.oldestValidPosition` is measured from the
    /// highest position ever WRITTEN, so after the rollback the destroyed
    /// positions are outside the row's read guard and `attendableViews()`
    /// TRAPS rather than handing back substituted KV — which is the guard
    /// doing its job, and why this control cannot be a fingerprint diff any
    /// more. So: the row's own report of the damage, pinned to exactly the
    /// one position a one-token overrun reaches, plus the slab byte behind
    /// it read straight from the pool.
    @Test func roundBeyondTheRingHeadroomDestroysHistory() throws {
        let window = 32
        let cfg = config()
        let kind = windowedKind(window)
        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)

        let ringPageCount = PagedKVPool.ringPageCount(window: window, config: cfg)
        let ringTokens = ringPageCount * cfg.pageSize
        let headroom = ringTokens - window
        #expect(
            headroom >= CBv2PagedSpeculation.maxSpeculativeSpan,
            "the ring must reserve at least one round of headroom")
        #expect(row.speculativeHeadroom == headroom, "the row reads the same headroom")

        fill(row, 0 ..< 104, chunk: cfg.maxPrefillChunk)
        let before = fingerprint(row)
        #expect(before.retainedCount == window)

        // One token past the headroom is enough: position `104 + headroom`
        // destroys `104 + headroom - ringTokens == 104 - window`, which is
        // the OLDEST still-attendable position. Written in `maxPrefillChunk`
        // pieces because `write` refuses a longer one — the overrun is in the
        // ROUND's length, not in any single write.
        let overrun = headroom + 1
        fill(row, 104 ..< (104 + overrun), chunk: cfg.maxPrefillChunk)
        row.rollback(overrun)

        #expect(row.absoluteOffset == before.absoluteOffset, "the counter still rewinds")
        #expect(row.retainedCount == before.retainedCount, "and the exposure is unchanged")

        let destroyed = before.absoluteOffset - window
        #expect(
            row.oldestValidPosition == destroyed + 1,
            """
            a \(overrun)-token round past a \(headroom)-token headroom left the row holding \
            everything from \(row.oldestValidPosition); position \(destroyed) should be gone. \
            Either the ring grew, or the round no longer writes through to the slabs.
            """)
        #expect(
            row.oldestValidPosition > row.absoluteOffset - row.retainedCount,
            "the row still exposes a window it no longer physically holds")

        // The slab byte behind that report: the destroyed slot holds the
        // round's token, not the pre-round one.
        let usurper = destroyed + ringTokens
        let (slotKeys, _) = backend.pool.gather(
            group: row.groupKey,
            pages: [row.table[(destroyed / cfg.pageSize) % ringPageCount]],
            firstSlot: destroyed % cfg.pageSize, count: 1)
        #expect(
            hostCopy(slotKeys) == hostCopy(coded(usurper ..< (usurper + 1)).0),
            """
            the slot of position \(destroyed) does not hold position \(usurper) — the round \
            did not write through to the slabs.
            """)
    }

    /// A row that finishes mid-round is torn down by the deferred-release
    /// fence and NEVER receives its `commitSpeculativeWrite`
    /// (`CBv2MTPRoundDriver`: "the whole state is released ... and this list
    /// is not touched"). Release therefore has to drain the queue itself, or
    /// the queued pages are orphaned — reserved, refcounted, and off the free
    /// list forever. Draining there is safe because releases only happen on
    /// the engine thread between host-synced steps.
    @Test func releaseMidRoundDrainsTheQueueInsteadOfOrphaningPages() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        let group = backend.pool.group(PagedKVGroupKey(kind))
        let emptyPool = fingerprint(backend.pool, PagedKVGroupKey(kind))

        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 128)
        let row = try #require(state[0] as? PagedSequenceKV)
        write(row, 0 ..< 30)
        row.beginSpeculativeWrite()
        write(row, 30 ..< 38)
        row.rollback(8)
        #expect(group.deferredFrees.count == 1, "a page is queued and the round never commits")

        backend.release(state)

        // The free list is a BAG, not a sequence: release drains the queue
        // first and then returns the table, so page 3 is pushed before 1 and
        // 2 and the stack ends [.., 4, 3, 1, 2] where allocation order would
        // have left [.., 4, 3, 2, 1]. Nothing depends on which free page is
        // handed out next, so the invariant is the multiset plus the
        // refcounts — which is what actually distinguishes a drained queue
        // from an orphan (page missing) or a double free (page duplicated,
        // refcount underflowed). Asserted here rather than
        // `PoolFingerprint ==`, which is order-sensitive and is kept exact in
        // `fullRowRoundDefersPageFreeToCommitAndRestoresFreeList`, where a
        // single pop/push does round-trip the order.
        let after = fingerprint(backend.pool, PagedKVGroupKey(kind))
        #expect(after.deferredFrees.isEmpty, "release drained the queue")
        #expect(after.freeList.sorted() == emptyPool.freeList.sorted(), "no page orphaned")
        #expect(
            after.freeList.count == Set(after.freeList).count,
            "no page freed twice")
        #expect(after.refCounts == emptyPool.refCounts, "refcounts fully unwound")
        #expect(after.pagesInUse == emptyPool.pagesInUse)
        #expect(backend.pool.bytesReserved == 0)
    }

    @Test func fullRowRoundDefersPageFreeToCommitAndRestoresFreeList() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 128)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let group = backend.pool.group(row.groupKey)

        write(row, 0 ..< 30)  // 2 pages
        #expect(row.table.count == 2)
        let poolBefore = fingerprint(backend.pool, row.groupKey)
        let tableBefore = row.table

        row.beginSpeculativeWrite()
        write(row, 30 ..< 38)  // crosses into a third page
        #expect(row.table.count == 3)
        let doomed = row.table[2]
        #expect(group.pagesInUse == poolBefore.pagesInUse + 1)

        row.rollback(8)
        #expect(row.table == tableBefore, "the speculative page leaves the table")
        #expect(group.deferredFrees == [doomed], "queued, not freed")
        #expect(
            group.pagesInUse == poolBefore.pagesInUse + 1,
            "an in-round free must not return the page — a lazy capture still names it")
        #expect(!group.freeList.contains(doomed), "it must not be reallocatable mid-round")
        #expect(group.refCounts[Int(doomed)] == 1)

        row.commitSpeculativeWrite()
        #expect(
            fingerprint(backend.pool, row.groupKey) == poolBefore,
            "commit restores the free list and refcounts exactly")

        // Contrast: OUTSIDE a transaction the free is immediate, so the
        // deferral is transaction-scoped rather than a blanket change.
        write(row, 30 ..< 38)
        #expect(group.pagesInUse == poolBefore.pagesInUse + 1)
        row.rollback(8)
        #expect(group.deferredFrees.isEmpty)
        #expect(fingerprint(backend.pool, row.groupKey) == poolBefore)
    }

    @Test func partiallyAcceptedRoundEqualsPlainDecodeShapedRow() throws {
        let window = 32
        let cfg = config()
        let kind = windowedKind(window)
        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let speculatedState = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        let plainState = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer {
            backend.release(speculatedState)
            backend.release(plainState)
        }
        let speculated = try #require(speculatedState[0] as? PagedSequenceKV)
        let plain = try #require(plainState[0] as? PagedSequenceKV)

        for row in [speculated, plain] { fill(row, 0 ..< 100, chunk: cfg.maxPrefillChunk) }

        // 1 target + 7 drafts written as one rectangular block, 3 rejected.
        speculated.beginSpeculativeWrite()
        write(speculated, 100 ..< 108)
        speculated.rollback(3)
        speculated.commitSpeculativeWrite()

        // MTP-off equivalent: the 5 accepted tokens, one decode at a time.
        for p in 100 ..< 105 { write(plain, p ..< (p + 1)) }

        #expect(speculated.absoluteOffset == plain.absoluteOffset)
        #expect(speculated.retainedCount == plain.retainedCount)
        #expect(speculated.retainedCount == window)
        #expect(speculated.table.count == plain.table.count)
        let (sk, sv) = speculated.attendableViews()
        let (pk, pv) = plain.attendableViews()
        #expect(hostCopy(sk) == hostCopy(pk), "accepted keys diverge from the plain run")
        #expect(hostCopy(sv) == hostCopy(pv), "accepted values diverge from the plain run")
    }

    /// A committed round must leave the row exposing exactly what the plain
    /// run exposes.
    ///
    /// This used to be a real risk and is now a structural property:
    /// `retainedCount` was `min(written, window - 1 + lastUpdateTokens)`, so
    /// a bulk round left it at `window - 1 + 8 == 39` unless `commit` reset
    /// the input, and forgetting that reset over-exposed the row against a
    /// plain decode of the same confirmed tokens. `retainedCount` is now
    /// `min(written, window)` in every phase, so there is no input to settle.
    /// The test stays because the OBSERVABLE it defends — a committed round
    /// is indistinguishable from the plain run — is what actually matters,
    /// and it would fail again the moment a phase-dependent exposure came
    /// back.
    @Test func committedBulkRoundDoesNotOverExposeRetainedCount() throws {
        let window = 32
        let cfg = config()
        let kind = windowedKind(window)
        let backend = try PagedKVBackend(layerKinds: [kind], config: cfg)
        let speculatedState = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        let plainState = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer {
            backend.release(speculatedState)
            backend.release(plainState)
        }
        let speculated = try #require(speculatedState[0] as? PagedSequenceKV)
        let plain = try #require(plainState[0] as? PagedSequenceKV)

        for row in [speculated, plain] { fill(row, 0 ..< 100, chunk: cfg.maxPrefillChunk) }

        // Every draft accepted: no rollback runs.
        speculated.beginSpeculativeWrite()
        write(speculated, 100 ..< 108)
        #expect(
            speculated.retainedCount == window,
            "a bulk round must not widen what the row exposes")
        speculated.commitSpeculativeWrite()

        for p in 100 ..< 108 { write(plain, p ..< (p + 1)) }

        #expect(speculated.retainedCount == window)
        #expect(speculated.retainedCount == plain.retainedCount)
        let (sk, sv) = speculated.attendableViews()
        let (pk, pv) = plain.attendableViews()
        #expect(hostCopy(sk) == hostCopy(pk))
        #expect(hostCopy(sv) == hostCopy(pv))
    }
}
