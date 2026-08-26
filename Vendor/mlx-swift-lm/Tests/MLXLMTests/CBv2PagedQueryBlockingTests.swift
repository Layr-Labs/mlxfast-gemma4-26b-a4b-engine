// CBv2PagedQueryBlockingTests.swift
//
// Two paged attention behaviours that share a theme — the paged cache must
// produce the SAME numbers however it chops the query axis up:
//
//   * WS-0.2p: `prefillAttend` attends a prompt chunk in query blocks
//     (`CBv2AttentionV1.queryBlockSize`, shared kill switch
//     `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK`), which bounds the materialized
//     score tensor at `[1, heads, block, visible]` instead of
//     `[1, heads, chunk, retained]`.
//   * WS-3.4: with `CBv2MTPRectangularSerializing` set, a rectangular
//     `[B, *, L > 1, *]` call is attended one query COLUMN at a time, each
//     column bit-identical to that column run as a standalone `L == 1`
//     decode.
//
// Both are checked against references that do not run the code under test:
// the whole-rectangle composed reference for blocking, and genuinely
// separate decode steps on an identically-primed twin row for rectangular
// verification.
//
// The blocking tests use chunks longer than the default block (128), so a
// default test process takes the BLOCKED path and
// `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK=0` takes the unblocked one. Both must
// agree with the reference, which is what makes the knob a kill switch
// rather than a behaviour change.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon
@testable import MLXLLM

@Suite("CBv2PagedQueryBlocking")
struct CBv2PagedQueryBlockingTests {

    // MARK: - Fixtures

    private let headDim = 64
    private let kvHeads = 2
    private let queryHeads = 4

    private var scale: Float { 1.0 / Float(headDim).squareRoot() }

    private func kind(window: Int?, sharesKVWithLayer: Int? = nil) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: window.map { .slidingWindow($0) } ?? .full,
            sharesKVWithLayer: sharesKVWithLayer,
            headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func config(maxPrefillChunk: Int, dtype: DType) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: 64 << 20, dtype: dtype,
            maxPrefillChunk: maxPrefillChunk, nominalMaxSequenceLength: 1024)
    }

    /// `[kvHeads, n, headDim]` K/V whose every element encodes the ABSOLUTE
    /// position, so attending the wrong slice of the gathered range is
    /// numerically loud instead of merely improbable.
    private func codedKV(_ positions: Range<Int>, dtype: DType) -> (MLXArray, MLXArray) {
        let n = positions.count
        var kflat = [Float](repeating: 0, count: kvHeads * n * headDim)
        var vflat = [Float](repeating: 0, count: kvHeads * n * headDim)
        var i = 0
        for h in 0 ..< kvHeads {
            for p in positions {
                for d in 0 ..< headDim {
                    kflat[i] = Float(p % 89) * 0.011 + Float(d) * 0.001 + Float(h) * 0.1
                    vflat[i] = Float(p % 89) * 0.021 - Float(d) * 0.0005 + Float(h) * 0.05
                    i += 1
                }
            }
        }
        return (
            MLXArray(kflat, [kvHeads, n, headDim]).asType(dtype),
            MLXArray(vflat, [kvHeads, n, headDim]).asType(dtype)
        )
    }

    /// Whole-rectangle reference: one composed attention call over ALL the
    /// retained keys with the explicit causal-and-window BOOL mask. No
    /// blocking, no slicing, no shared helper with `PagedLayerCache`.
    private func wholeRectangle(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        qStart: Int, kStart: Int, window: Int?, isBidirectional: Bool = false
    ) -> MLXArray {
        let qEnd = qStart + queries.dim(2)
        let kEnd = kStart + keys.dim(2)
        let qpos = MLXArray(Int32(qStart) ..< Int32(qEnd)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(kStart) ..< Int32(kEnd)).expandedDimensions(axis: 0)
        var mask = kpos .<= qpos
        if let window = window {
            mask = mask & (kpos .> (qpos - Int32(window)))
        }
        if isBidirectional {
            var reverse = (kpos .>= qpos) .&& (kpos .>= Int32(qStart))
            if let window = window {
                reverse = reverse .&& (kpos .< (qpos + Int32(window)))
            }
            mask = mask .|| reverse
        }
        return PagedAttentionReference.composedAttention(
            queries: queries, keys: keys, values: values, scale: scale, boolMask: mask)
    }

    private func assertClose(
        _ got: MLXArray, _ want: MLXArray, rtol: Double = 1e-3, atol: Double = 1e-4,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        let delta = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
        #expect(
            allClose(got, want, rtol: rtol, atol: atol).item(Bool.self),
            "max |delta| = \(delta)",
            sourceLocation: sourceLocation)
    }

    private func assertIdentical(
        _ got: MLXArray, _ want: MLXArray,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        let delta = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
        #expect(
            arrayEqual(got, want).item(Bool.self),
            "columns are not bit-identical: max |delta| = \(delta)",
            sourceLocation: sourceLocation)
    }

    // MARK: - WS-0.2p: query-blocked prefill

    /// The default block is 128, so a 192-token chunk is blocked 128 + 64.
    /// A full-attention layer's blocks all start at gathered column 0, which
    /// is exactly why the gather must stay hoisted — and why a bad
    /// `visibleEnd` (the one bound that DOES move) shows up here.
    @Test func blockedPrefillMatchesWholeRectangleOnFullAttention() throws {
        let layer = kind(window: nil)
        let chunk = 192
        let dtype = DType.float32
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
        let cache = backend.makeLayerCaches()[0]
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: chunk, maxLength: 512)
        defer { backend.release(state) }
        cache.setRows([state[0]!])

        let (k, v) = codedKV(0 ..< chunk, dtype: dtype)
        let queries = MLXRandom.normal([1, queryHeads, chunk, headDim], dtype: dtype)
        let got = cache.updateAndAttend(
            queries: queries,
            keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)

        assertClose(
            got,
            wholeRectangle(
                queries: queries,
                keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
                qStart: 0, kStart: 0, window: nil))
    }

    @Test func gemma4AllModePagedPrefillAttendsFutureKeys() throws {
        let json = """
            {
                "model_type": "gemma4_text",
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "head_dim": 64,
                "global_head_dim": 64,
                "num_key_value_heads": 2,
                "num_kv_shared_layers": 0,
                "layer_types": ["full_attention"],
                "use_bidirectional_attention": "all"
            }
            """
        let modelConfig = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        let layer = try #require(modelConfig.cbv2LayerKinds.first)
        #expect(layer.isBidirectional)

        let chunk = 192
        let dtype = DType.float32
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
        let cache = backend.makeLayerCaches()[0]
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: chunk, maxLength: 512)
        defer { backend.release(state) }
        cache.setRows([state[0]!])

        let (k, v) = codedKV(0 ..< chunk, dtype: dtype)
        let queries = MLXArray.zeros([1, queryHeads, chunk, headDim], dtype: dtype)
        let keys = k.expandedDimensions(axis: 0)
        let values = v.expandedDimensions(axis: 0)
        let got = cache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        let bidirectional = wholeRectangle(
            queries: queries, keys: keys, values: values,
            qStart: 0, kStart: 0, window: nil, isBidirectional: true)
        let causal = wholeRectangle(
            queries: queries, keys: keys, values: values,
            qStart: 0, kStart: 0, window: nil)

        assertClose(got, bidirectional)
        let causalDelta = abs(got - causal).max().item(Float.self)
        #expect(causalDelta > 0.01, "fixture must distinguish all-mode from causal attention")
    }

    /// Sliding window, and the WS-1.2 pin on gather ORDER: consecutive
    /// blocks' visible spans overlap by `window - 1`, so `visibleStart` is
    /// non-trivial and a block that mis-slices the keys attends another
    /// query's window.
    ///
    /// `prefillAttend` reads its keys from the view assembled BEFORE the
    /// chunk was written, so the only ring capacity it needs is
    /// `window - 1` — asserted below. A post-write gather would instead ask
    /// for `window - 1 + chunk` (447 here), the term that keeps
    /// `maxPrefillChunk` inside `ringPageCount`; once that term goes, a
    /// post-write gather aborts the process on `gatherRange`'s eviction
    /// precondition for any windowed chunk past `pageSize + 1` tokens.
    @Test func blockedPrefillMatchesWholeRectangleOnSlidingWindow() throws {
        let window = 256
        let layer = kind(window: window)
        let chunk = 192
        let dtype = DType.float32
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
        let ring = PagedKVPool.ringPageCount(window: window, config: backend.pool.config)
            * backend.pool.config.pageSize
        #expect(
            ring >= window - 1,
            "the pre-write gather needs exactly this much ring and no more")
        let cache = backend.makeLayerCaches()[0]
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: chunk, maxLength: 1024)
        defer { backend.release(state) }
        cache.setRows([state[0]!])

        let (k, v) = codedKV(0 ..< chunk, dtype: dtype)
        let queries = MLXRandom.normal([1, queryHeads, chunk, headDim], dtype: dtype)
        let got = cache.updateAndAttend(
            queries: queries,
            keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)

        assertClose(
            got,
            wholeRectangle(
                queries: queries,
                keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
                qStart: 0, kStart: 0, window: window))
    }

    /// The case that pins `historyCount`: a chunk whose queries are the
    /// trailing columns of a longer key range, with the window floor above
    /// the row's base so the history is CLAMPED. Every block bound shifts by
    /// the history, and an off-by-history bound silently attends the wrong
    /// absolute positions while still passing the single-chunk tests above.
    @Test func blockedPrefillHonoursPriorHistory() throws {
        let window = 256
        let layer = kind(window: window)
        let chunk = 192
        let priorChunks = 2
        let dtype = DType.float32
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
        let cache = backend.makeLayerCaches()[0]
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: chunk * (priorChunks + 1), maxLength: 1024)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        cache.setRows([row])

        for index in 0 ..< priorChunks {
            let (pk, pv) = codedKV((index * chunk) ..< ((index + 1) * chunk), dtype: dtype)
            _ = cache.updateAndAttend(
                queries: MLXRandom.normal([1, queryHeads, chunk, headDim], dtype: dtype),
                keys: pk.expandedDimensions(axis: 0), values: pv.expandedDimensions(axis: 0),
                scale: scale, sinks: nil)
        }

        let qStart = priorChunks * chunk
        #expect(row.absoluteOffset == qStart)
        let (k1, v1) = codedKV(qStart ..< (qStart + chunk), dtype: dtype)
        let queries = MLXRandom.normal([1, queryHeads, chunk, headDim], dtype: dtype)
        let got = cache.updateAndAttend(
            queries: queries,
            keys: k1.expandedDimensions(axis: 0), values: v1.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)

        // The assembled view is `gather([qStart - window + 1, qStart)) ++ chunk`,
        // clamped at the row's base.
        let kStart = max(0, qStart - window + 1)
        #expect(kStart > 0, "test premise: the window floor clamps above the row base")
        let (kr, vr) = codedKV(kStart ..< (qStart + chunk), dtype: dtype)
        assertClose(
            got,
            wholeRectangle(
                queries: queries,
                keys: kr.expandedDimensions(axis: 0), values: vr.expandedDimensions(axis: 0),
                qStart: qStart, kStart: kStart, window: window))
    }

    /// Softcapped configs take the composed path, and they must be blocked
    /// too — otherwise a capped model keeps the full unblocked score-tensor
    /// peak that WS-0.2p exists to remove.
    @Test func blockedPrefillMatchesWholeRectangleWithSoftcap() throws {
        let layer = kind(window: nil)
        let chunk = 192
        let softcap: Float = 30
        let dtype = DType.float32
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
        let cache = backend.makeLayerCaches(attentionSoftcap: softcap)[0]
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: chunk, maxLength: 512)
        defer { backend.release(state) }
        cache.setRows([state[0]!])

        let (k, v) = codedKV(0 ..< chunk, dtype: dtype)
        let queries = MLXRandom.normal([1, queryHeads, chunk, headDim], dtype: dtype)
        let got = cache.updateAndAttend(
            queries: queries,
            keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)

        let qpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 0)
        let want = PagedAttentionReference.composedAttention(
            queries: queries,
            keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
            scale: scale, boolMask: kpos .<= qpos, sinks: nil, softcap: softcap)
        assertClose(got, want)
    }

    /// A chunk at or below the block size must take the single unblocked
    /// call — the loop is an optimisation, not a new numerics path, and the
    /// short-chunk case is the one every existing paged test exercises.
    @Test func shortChunksKeepTheSingleUnblockedCall() throws {
        let layer = kind(window: nil)
        let chunk = min(64, max(1, CBv2AttentionV1.queryBlockSize))
        #expect(
            !CBv2AttentionV1.shouldBlockQueries(chunk),
            "premise: a \(chunk)-token chunk is at or below the configured block")
        let dtype = DType.float32
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
        let cache = backend.makeLayerCaches()[0]
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: chunk, maxLength: 256)
        defer { backend.release(state) }
        cache.setRows([state[0]!])

        let (k, v) = codedKV(0 ..< chunk, dtype: dtype)
        let queries = MLXRandom.normal([1, queryHeads, chunk, headDim], dtype: dtype)
        let got = cache.updateAndAttend(
            queries: queries,
            keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)
        assertClose(
            got,
            wholeRectangle(
                queries: queries,
                keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
                qStart: 0, kStart: 0, window: nil))
    }

    // MARK: - WS-3.4: rectangular MTP verification

    /// `PagedLayerCache` must CONFORM, not just behave: the engine's
    /// rectangular arm gates on the conformance and silently degrades to the
    /// serial oracle without it.
    @Test func pagedCacheConformsToRectangularSerializing() throws {
        let layer = kind(window: nil)
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: 16, dtype: .float16))
        let cache: any CBv2AttendingLayerCache = backend.makeLayerCaches()[0]
        let serializing = try #require(cache as? CBv2MTPRectangularSerializing)
        #expect(!serializing.mtpSerializesRectangularAttention, "off by default")
        serializing.mtpSerializesRectangularAttention = true
        #expect(serializing.mtpSerializesRectangularAttention)
    }

    /// A KV-OWNING layer under the flag: `[B, *, 1 + k, *]` in one call must
    /// equal the same columns fed as separate decode steps to an
    /// identically-primed twin. Without the flag this call would trap on
    /// `precondition(b == 1)`.
    @Test func rectangularOwningLayerEqualsSerialDecodeColumns() throws {
        let layer = kind(window: 8)
        let prime = 16
        let columns = 3
        let batch = 2
        let dtype = DType.float16
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: prime, dtype: dtype))

        func makeRows() throws -> ([CBv2SequenceKV?], [PagedSequenceKV]) {
            var states: [CBv2SequenceKV?] = []
            var rows: [PagedSequenceKV] = []
            for _ in 0 ..< batch {
                let state = try backend.makeSequenceState(
                    layerKinds: [layer], promptLength: prime, maxLength: 128)
                let row = try #require(state[0] as? PagedSequenceKV)
                let (k, v) = codedKV(0 ..< prime, dtype: dtype)
                row.write(keys: k, values: v)
                states.append(state[0])
                rows.append(row)
            }
            return (states, rows)
        }

        let (statesA, rowsA) = try makeRows()
        let (statesB, rowsB) = try makeRows()
        defer {
            backend.release(statesA)
            backend.release(statesB)
        }

        let queries = MLXRandom.normal([batch, queryHeads, columns, headDim], dtype: dtype)
        let (ck, cv) = codedKV(prime ..< (prime + columns), dtype: dtype)
        // Same column values for every row in the batch.
        let keys = stacked([MLXArray](repeating: ck, count: batch), axis: 0)
        let values = stacked([MLXArray](repeating: cv, count: batch), axis: 0)

        let rectangular = backend.makeLayerCaches()[0]
        rectangular.setRows(rowsA)
        rectangular.mtpSerializesRectangularAttention = true
        let got = rectangular.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        rectangular.mtpSerializesRectangularAttention = false

        let serial = backend.makeLayerCaches()[0]
        serial.setRows(rowsB)
        var wantColumns: [MLXArray] = []
        for t in 0 ..< columns {
            wantColumns.append(
                serial.updateAndAttend(
                    queries: queries[0..., 0..., t ..< (t + 1), 0...],
                    keys: keys[0..., 0..., t ..< (t + 1), 0...],
                    values: values[0..., 0..., t ..< (t + 1), 0...],
                    scale: scale, sinks: nil))
        }
        assertIdentical(got, concatenated(wantColumns, axis: 2))

        // Both paths advanced every row by exactly the column count.
        #expect(rowsA.map { $0.absoluteOffset } == rowsB.map { $0.absoluteOffset })
        #expect(rowsA.allSatisfy { $0.absoluteOffset == prime + columns })
        assertIdentical(rectangular.positionOffsets, serial.positionOffsets)
    }

    /// The delicate half: a KV-BORROWING layer runs AFTER the source layer
    /// advanced every row by L, so `decodeAttendRange` alone resolves every
    /// column to the last one. Only an explicit per-column `qPos` gives each
    /// borrowed column its own visible window — a windowed source makes the
    /// difference show up in the range START, not just its length.
    @Test func rectangularBorrowingLayerEqualsSerialDecodeColumns() throws {
        let window = 8
        let kinds = [kind(window: window), kind(window: window, sharesKVWithLayer: 0)]
        let prime = 16
        let columns = 3
        let batch = 2
        let dtype = DType.float16
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: prime, dtype: dtype))

        func makeRows() throws -> ([[CBv2SequenceKV?]], [PagedSequenceKV]) {
            var states: [[CBv2SequenceKV?]] = []
            var rows: [PagedSequenceKV] = []
            for _ in 0 ..< batch {
                let state = try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: prime, maxLength: 128)
                #expect(state[1] == nil, "KV-shared layers own no storage")
                let row = try #require(state[0] as? PagedSequenceKV)
                let (k, v) = codedKV(0 ..< prime, dtype: dtype)
                row.write(keys: k, values: v)
                states.append(state)
                rows.append(row)
            }
            return (states, rows)
        }

        let (statesA, rowsA) = try makeRows()
        let (statesB, rowsB) = try makeRows()
        defer {
            for state in statesA + statesB { backend.release(state) }
        }

        let sourceQueries = MLXRandom.normal([batch, queryHeads, columns, headDim], dtype: dtype)
        let sharedQueries = MLXRandom.normal([batch, queryHeads, columns, headDim], dtype: dtype)
        let (ck, cv) = codedKV(prime ..< (prime + columns), dtype: dtype)
        let keys = stacked([MLXArray](repeating: ck, count: batch), axis: 0)
        let values = stacked([MLXArray](repeating: cv, count: batch), axis: 0)

        let rectangular = backend.makeLayerCaches()
        rectangular[0].setRows(rowsA)
        for cache in rectangular { cache.mtpSerializesRectangularAttention = true }
        _ = rectangular[0].updateAndAttend(
            queries: sourceQueries, keys: keys, values: values, scale: scale, sinks: nil)
        let got = rectangular[1].attendBorrowing(
            source: rectangular[0], queries: sharedQueries, scale: scale, sinks: nil)
        for cache in rectangular { cache.mtpSerializesRectangularAttention = false }

        let serial = backend.makeLayerCaches()
        serial[0].setRows(rowsB)
        var wantColumns: [MLXArray] = []
        for t in 0 ..< columns {
            _ = serial[0].updateAndAttend(
                queries: sourceQueries[0..., 0..., t ..< (t + 1), 0...],
                keys: keys[0..., 0..., t ..< (t + 1), 0...],
                values: values[0..., 0..., t ..< (t + 1), 0...],
                scale: scale, sinks: nil)
            wantColumns.append(
                serial[1].attendBorrowing(
                    source: serial[0],
                    queries: sharedQueries[0..., 0..., t ..< (t + 1), 0...],
                    scale: scale, sinks: nil))
        }
        assertIdentical(got, concatenated(wantColumns, axis: 2))
    }

    /// The FLAG is the router, and clearing it must route a rectangular
    /// `[B, L > 1]` call to the PROMPT-CHUNK path — not to the per-column
    /// decode path, and (since WS-2.1) not to a trap either.
    ///
    /// History, because this test changed shape rather than meaning. It used
    /// to assert `!bank.supportsPackedPrefill`, and its doc claimed that
    /// pinned "a rectangular prompt chunk is still refused" — but it never
    /// issued one, or called `updateAndAttend` at all. The closed capability
    /// gate was a PROXY: while `PagedLayerCache` had a `precondition(b == 1)`
    /// on the prompt branch, the only thing keeping the engine from tripping
    /// it was the bank refusing to vouch. The invariant actually being
    /// defended was "the cache must not claim a capability it does not
    /// implement". That invariant is unchanged; paged now implements packed
    /// prefill, so the claim flips WITH the implementation, and the routing
    /// the doc always described is asserted directly here for the first
    /// time. Bit-identity of a packed row against the same row run alone —
    /// what makes the claim TRUE rather than merely asserted — is covered by
    /// `CBv2PagedPackedSpanTests.packedPrefillIsBitIdenticalToUnpackedPerRow`
    /// and `.packedPrefillHonoursPerRowOffsetsOnSlidingWindow`.
    @Test func clearedFlagRoutesRectangularCallsToPackedPrefill() throws {
        let layer = kind(window: nil)
        let chunk = 12
        let batch = 2
        let dtype = DType.float16
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: 16, dtype: dtype))

        func makeRows() throws -> ([CBv2SequenceKV?], [PagedSequenceKV]) {
            var states: [CBv2SequenceKV?] = []
            var rows: [PagedSequenceKV] = []
            for _ in 0 ..< batch {
                let state = try backend.makeSequenceState(
                    layerKinds: [layer], promptLength: chunk, maxLength: 64)
                states.append(state[0])
                rows.append(try #require(state[0] as? PagedSequenceKV))
            }
            return (states, rows)
        }
        let (promptStates, promptRows) = try makeRows()
        let (columnStates, columnRows) = try makeRows()
        defer {
            backend.release(promptStates)
            backend.release(columnStates)
        }

        let queries = MLXRandom.normal([batch, queryHeads, chunk, headDim], dtype: dtype)
        let (ck, cv) = codedKV(0 ..< chunk, dtype: dtype)
        let keys = stacked([MLXArray](repeating: ck, count: batch), axis: 0)
        let values = stacked([MLXArray](repeating: cv, count: batch), axis: 0)

        // Flag CLEARED: the prompt-chunk path. It attends gathered pages
        // through SDPA and never builds a device block table, because only
        // `dispatchDecode` needs one.
        let prompt = backend.makeLayerCaches()[0]
        #expect(!prompt.mtpSerializesRectangularAttention, "off by default")
        prompt.setRows(promptRows)
        _ = prompt.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        #expect(
            prompt.tablesRebuildCount == 0,
            "the prompt-chunk path must not touch the decode kernel's block tables")

        // Flag SET, same shape: the per-column decode path, which does.
        let columns = backend.makeLayerCaches()[0]
        columns.setRows(columnRows)
        columns.mtpSerializesRectangularAttention = true
        _ = columns.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        columns.mtpSerializesRectangularAttention = false
        #expect(
            columns.tablesRebuildCount > 0,
            "premise: the per-column path DOES build block tables, so the check above discriminates the two routes rather than being vacuous")

        // Both routes advanced every row by the full chunk.
        #expect(promptRows.allSatisfy { $0.absoluteOffset == chunk })
        #expect(columnRows.allSatisfy { $0.absoluteOffset == chunk })

        // And the capability claim now matches the implementation.
        #expect(
            CBv2LayerCacheBank(caches: [prompt]).supportsPackedPrefill,
            "paged implements packed prefill, so the bank must vouch")
    }
}

/// The MEMORY bound WS-0.2p buys, asserted directly on the block arithmetic.
///
/// The numerics tests above prove blocking is equivalent; they cannot prove
/// it is bounded, because the score tensor is never a value anyone can
/// inspect. These do, at gemma-4's real geometry, and they are pure host
/// arithmetic — no pool, no MLX, no allocation.
///
/// Measured shapes these pin (chunk 512, `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK`
/// at its default 128):
///   sliding, window 1024, 8 kv heads, steady state
///       unblocked  [1, 8, 512, 1535]   blocked  4x [1, 8, 128, 1151]
///   full, 2 kv heads, 8192 tokens of history
///       unblocked  [1, 2, 512, 8704]   blocked  4x [1, 2, 128, <=8704]
@Suite("CBv2PagedQueryBlockBounds")
struct CBv2PagedQueryBlockBoundsTests {

    private func spans(historyCount: Int, chunk: Int, blockSize: Int, window: Int?) -> [(
        queries: Int, keys: Int
    )] {
        var result: [(queries: Int, keys: Int)] = []
        var offset = 0
        while offset < chunk {
            let count = min(blockSize, chunk - offset)
            let bounds = CBv2AttentionV1.queryBlockBounds(
                historyCount: historyCount, offset: offset, count: count, window: window)
            result.append((queries: count, keys: bounds.visibleEnd - bounds.visibleStart))
            offset += count
        }
        return result
    }

    /// A sliding layer's per-block key span saturates at
    /// `window - 1 + blockSize` and is INDEPENDENT of the chunk length —
    /// that independence is the whole point, because it is what lets
    /// `prefillChunkSize` grow without the activation reserve becoming a
    /// lie. The unblocked span grows with the chunk instead.
    @Test func slidingBlockSpanIsIndependentOfChunkLength() {
        let window = 1024
        let blockSize = 128
        for chunk in [256, 512, 1024, 2048] {
            let blocks = spans(
                historyCount: window - 1, chunk: chunk, blockSize: blockSize, window: window)
            #expect(blocks.count == chunk / blockSize)
            #expect(
                blocks.allSatisfy { $0.queries == blockSize && $0.keys == window - 1 + blockSize },
                "chunk \(chunk) produced \(blocks)")
        }
        // The single unblocked call the loop replaces grows with the chunk.
        for chunk in [256, 512, 1024, 2048] {
            let whole = spans(
                historyCount: window - 1, chunk: chunk, blockSize: chunk, window: window)
            #expect(whole.count == 1)
            #expect(whole.first?.queries == chunk)
            #expect(whole.first?.keys == window - 1 + chunk)
        }
    }

    /// A full-attention layer's blocks all start at gathered column 0 (every
    /// query sees the whole history), so blocking caps the QUERY extent
    /// only — exactly the `chunk / blockSize` factor, and exactly why the
    /// gather must not be repeated per block.
    @Test func fullAttentionBlocksShareTheWholeHistory() {
        let history = 8192
        let chunk = 512
        let blockSize = 128
        let blocks = spans(
            historyCount: history, chunk: chunk, blockSize: blockSize, window: nil)
        #expect(blocks.count == 4)
        #expect(blocks.allSatisfy { $0.queries == blockSize })
        #expect(blocks.map { $0.keys } == [8320, 8448, 8576, 8704])
        let peak = blocks.map { $0.queries * $0.keys }.max() ?? 0
        #expect(
            (chunk * (history + chunk)) / peak == 4,
            "the unblocked score tensor is exactly chunk/blockSize times the blocked peak")
    }

    /// Every block's queries must be the TRAILING entries of its key span —
    /// the invariant the causal mask and `qpos`/`kpos` slicing both assume.
    @Test func queriesAreAlwaysTheTrailingKeys() {
        for window in [nil, 8, 64, 1024] as [Int?] {
            for historyCount in [0, 1, 63, 1023, 8192] {
                for offset in [0, 1, 128, 384] where offset < 512 {
                    let count = min(128, 512 - offset)
                    let bounds = CBv2AttentionV1.queryBlockBounds(
                        historyCount: historyCount, offset: offset, count: count, window: window)
                    #expect(bounds.visibleEnd == historyCount + offset + count)
                    #expect(bounds.visibleStart >= 0)
                    #expect(
                        bounds.visibleEnd - bounds.visibleStart >= count,
                        "a block must at least see its own queries")
                }
            }
        }
    }
}
