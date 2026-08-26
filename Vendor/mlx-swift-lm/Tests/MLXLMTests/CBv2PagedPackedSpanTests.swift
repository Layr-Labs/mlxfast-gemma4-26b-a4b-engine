// CBv2PagedPackedSpanTests.swift
//
// The two WS-2 capabilities the paged cache now AFFIRMS, and the proof that
// each affirmation is honoured rather than merely declared:
//
//   * WS-2.1 `CBv2PackedPrefillCapableCache.keepsRowsIndependentWhenPacked`
//     — a rectangular `[B > 1, L > 1]` prompt pass is per-row work over
//     per-row storage, so every packed row is BIT-IDENTICAL to that row run
//     alone. Asserted with `arrayEqual`, never `allClose`: a cross-row leak
//     is a few keys out of hundreds and lands well inside any tolerance.
//   * WS-2.2 `CBv2MultimodalSpanCapableCache.honorsSpanMaskContexts` — a
//     bound `CBv2SpanChunkContext` reaches the mask, as the same
//     bidirectional-within-block overlay the contiguous backend applies.
//
// Cross-row independence is attacked from three sides, because "no
// contamination was observed" is not the claim:
//   1. Rows carrying DELIBERATELY DIFFERENT content at the SAME absolute
//      positions — identical geometry, so a mis-sliced key or a mask built
//      from the wrong row's offsets lands on a real batchmate value instead
//      of out of bounds.
//   2. Rows at DIFFERENT pre-existing offsets, on a sliding-window layer,
//      where each row's visible span starts somewhere else.
//   3. INVARIANCE: row 0's output is unchanged when a batchmate's content is
//      replaced wholesale, while that batchmate's own output does change.
//
// Everything runs on tiny synthetic K/V through the real `PagedKVBackend`.
// No checkpoints, no model.

import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedPackedSpan")
struct CBv2PagedPackedSpanTests {

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
            capacityBytes: 256 << 20, dtype: dtype,
            maxPrefillChunk: maxPrefillChunk, nominalMaxSequenceLength: 4096)
    }

    /// `[kvHeads, n, headDim]` K/V in which BOTH the absolute position and
    /// the owning row are encoded in every element. `tag` is the row's
    /// signature: two rows with different tags hold different numbers at
    /// every one of their shared positions, so reading a batchmate's key is
    /// numerically loud rather than statistically improbable.
    private func codedKV(_ positions: Range<Int>, tag: Int, dtype: DType) -> (MLXArray, MLXArray)
    {
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
            MLXArray(kflat, [kvHeads, n, headDim]).asType(dtype),
            MLXArray(vflat, [kvHeads, n, headDim]).asType(dtype)
        )
    }

    private func assertIdentical(
        _ got: MLXArray, _ want: MLXArray, _ what: String = "arrays",
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

    private func assertClose(
        _ got: MLXArray, _ want: MLXArray, _ what: String = "arrays",
        rtol: Double = 1e-4, atol: Double = 1e-5,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, "\(what): shape", sourceLocation: sourceLocation)
        guard got.shape == want.shape else { return }
        let delta = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
        #expect(
            allClose(got, want, rtol: rtol, atol: atol).item(Bool.self),
            "\(what): max |delta| = \(delta)",
            sourceLocation: sourceLocation)
    }

    /// One paged row, primed with `priming` tokens of its own coded K/V so
    /// packed rows can start at DIFFERENT absolute offsets. Priming is
    /// written in `maxPrefillChunk`-sized pieces: a windowed row refuses a
    /// single write larger than that, because its ring cannot hold it.
    private func makeRow(
        backend: PagedKVBackend, layer: CBv2LayerKind, tag: Int, priming: Int,
        maxLength: Int, dtype: DType
    ) throws -> (state: [CBv2SequenceKV?], row: PagedSequenceKV) {
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: maxLength, maxLength: maxLength)
        let row = try #require(state[0] as? PagedSequenceKV)
        let step = backend.pool.config.maxPrefillChunk
        var written = 0
        while written < priming {
            let count = min(step, priming - written)
            let (k, v) = codedKV(written ..< (written + count), tag: tag, dtype: dtype)
            row.write(keys: k, values: v)
            written += count
        }
        return (state, row)
    }

    // MARK: - WS-2.1: packed prefill is per-row work

    /// The headline. Four rows, SAME absolute positions, DELIBERATELY
    /// DIFFERENT content at every one of those positions: if any row's
    /// gather, write, or mask reached into a batchmate, the numbers move.
    /// `arrayEqual`, because a leak of a few keys out of 192 is small.
    ///
    /// The chunk is 192 tokens, above the default 128 query block, so the
    /// packed rows are ALSO exercising WS-0.2p's per-row block loop — the
    /// two features have to compose, and the per-row `historyCount` is the
    /// obvious thing a packed port gets wrong.
    @Test func packedPrefillIsBitIdenticalToUnpackedPerRow() throws {
        let layer = kind(window: nil)
        let chunk = 192
        let batch = 4
        let dtype = DType.float32
        #expect(
            CBv2AttentionV1.shouldBlockQueries(chunk)
                || CBv2AttentionV1.queryBlockSize == 0,
            "premise: a \(chunk)-token chunk is blocked unless the kill switch is set")

        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))

        var packedStates: [[CBv2SequenceKV?]] = []
        var packedRows: [PagedSequenceKV] = []
        var referenceStates: [[CBv2SequenceKV?]] = []
        var referenceRows: [PagedSequenceKV] = []
        for tag in 0 ..< batch {
            let a = try makeRow(
                backend: backend, layer: layer, tag: tag, priming: 0,
                maxLength: chunk, dtype: dtype)
            let b = try makeRow(
                backend: backend, layer: layer, tag: tag, priming: 0,
                maxLength: chunk, dtype: dtype)
            packedStates.append(a.state)
            packedRows.append(a.row)
            referenceStates.append(b.state)
            referenceRows.append(b.row)
        }
        defer { for s in packedStates + referenceStates { backend.release(s) } }

        // Per-row queries and per-row K/V — nothing is shared across rows.
        var rowQueries: [MLXArray] = []
        var rowKeys: [MLXArray] = []
        var rowValues: [MLXArray] = []
        for tag in 0 ..< batch {
            let (k, v) = codedKV(0 ..< chunk, tag: tag, dtype: dtype)
            rowKeys.append(k.expandedDimensions(axis: 0))
            rowValues.append(v.expandedDimensions(axis: 0))
            rowQueries.append(
                MLXRandom.normal(
                    [1, queryHeads, chunk, headDim], key: MLXRandom.key(UInt64(9_100 + tag))
                ).asType(dtype))
        }

        let packedCache = backend.makeLayerCaches()[0]
        packedCache.setRows(packedRows)
        let packed = packedCache.updateAndAttend(
            queries: concatenated(rowQueries, axis: 0),
            keys: concatenated(rowKeys, axis: 0),
            values: concatenated(rowValues, axis: 0),
            scale: scale, sinks: nil)
        #expect(packed.shape == [batch, queryHeads, chunk, headDim])

        for tag in 0 ..< batch {
            // A FRESH cache per row: "run alone" must mean alone, not
            // "last row bound to a cache that saw the others".
            let solo = backend.makeLayerCaches()[0]
            solo.setRows([referenceRows[tag]])
            let want = solo.updateAndAttend(
                queries: rowQueries[tag], keys: rowKeys[tag], values: rowValues[tag],
                scale: scale, sinks: nil)
            assertIdentical(
                packed[tag ..< (tag + 1)], want, "packed row \(tag) vs the same row run alone")
        }

        // The KV each row LEFT BEHIND must match too — a packed write that
        // landed a batchmate's tokens in this row's pages would still
        // produce the right attention output for this step and poison the
        // next one.
        for tag in 0 ..< batch {
            #expect(packedRows[tag].absoluteOffset == chunk)
            #expect(referenceRows[tag].absoluteOffset == chunk)
            let got = packedRows[tag].snapshot()
            let want = referenceRows[tag].snapshot()
            #expect(got.offset == want.offset)
            assertIdentical(got.keys, want.keys, "packed row \(tag) retained keys")
            assertIdentical(got.values, want.values, "packed row \(tag) retained values")
        }
    }

    /// Sliding window, and rows at DIFFERENT pre-existing offsets — the
    /// realistic packed group, since `EngineLoopV2` groups prompt rows by
    /// chunk LENGTH only, never by how far along they are. Each row's
    /// gathered history and window floor are therefore different, which is
    /// exactly what a shared `historyCount` or a batch-wide `qStart` would
    /// silently flatten.
    @Test func packedPrefillHonoursPerRowOffsetsOnSlidingWindow() throws {
        let window = 128
        let layer = kind(window: window)
        let chunk = 96
        let primings = [0, 64, 208]
        let dtype = DType.float32
        let maxLength = 512

        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))

        var packedStates: [[CBv2SequenceKV?]] = []
        var packedRows: [PagedSequenceKV] = []
        var referenceStates: [[CBv2SequenceKV?]] = []
        var referenceRows: [PagedSequenceKV] = []
        for (tag, priming) in primings.enumerated() {
            let a = try makeRow(
                backend: backend, layer: layer, tag: tag, priming: priming,
                maxLength: maxLength, dtype: dtype)
            let b = try makeRow(
                backend: backend, layer: layer, tag: tag, priming: priming,
                maxLength: maxLength, dtype: dtype)
            packedStates.append(a.state)
            packedRows.append(a.row)
            referenceStates.append(b.state)
            referenceRows.append(b.row)
        }
        defer { for s in packedStates + referenceStates { backend.release(s) } }

        #expect(
            packedRows.map { $0.absoluteOffset } == primings,
            "premise: the packed rows start at three different offsets")
        #expect(
            primings.contains(where: { $0 >= window }),
            "premise: at least one row's window floor is above its base, so its history clamps")

        var rowQueries: [MLXArray] = []
        var rowKeys: [MLXArray] = []
        var rowValues: [MLXArray] = []
        for (tag, priming) in primings.enumerated() {
            let (k, v) = codedKV(priming ..< (priming + chunk), tag: tag, dtype: dtype)
            rowKeys.append(k.expandedDimensions(axis: 0))
            rowValues.append(v.expandedDimensions(axis: 0))
            rowQueries.append(
                MLXRandom.normal(
                    [1, queryHeads, chunk, headDim], key: MLXRandom.key(UInt64(9_200 + tag))
                ).asType(dtype))
        }

        let packedCache = backend.makeLayerCaches()[0]
        packedCache.setRows(packedRows)
        let packed = packedCache.updateAndAttend(
            queries: concatenated(rowQueries, axis: 0),
            keys: concatenated(rowKeys, axis: 0),
            values: concatenated(rowValues, axis: 0),
            scale: scale, sinks: nil)

        for tag in primings.indices {
            let solo = backend.makeLayerCaches()[0]
            solo.setRows([referenceRows[tag]])
            let want = solo.updateAndAttend(
                queries: rowQueries[tag], keys: rowKeys[tag], values: rowValues[tag],
                scale: scale, sinks: nil)
            assertIdentical(
                packed[tag ..< (tag + 1)], want,
                "packed row \(tag) (offset \(primings[tag])) vs the same row run alone")
        }

        // Every row advanced by exactly the chunk, and the batch-wide
        // on-device offset vector tracked all three independently.
        #expect(packedRows.map { $0.absoluteOffset } == primings.map { $0 + chunk })
        assertIdentical(
            packedCache.positionOffsets,
            MLXArray(primings.map { Int32($0 + chunk) }),
            "packed positionOffsets")
    }

    /// INVARIANCE, the property "bit-identical to unpacked" implies but does
    /// not display: replacing a batchmate's content wholesale must not move
    /// row 0 by a single bit, while it MUST move that batchmate — otherwise
    /// the test could pass with an attention that ignores its inputs.
    @Test func packedRowIsInvariantToBatchmateContent() throws {
        let layer = kind(window: nil)
        let chunk = 160
        let dtype = DType.float32
        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))

        let queries0 = MLXRandom.normal(
            [1, queryHeads, chunk, headDim], key: MLXRandom.key(9_301)
        ).asType(dtype)
        let queries1 = MLXRandom.normal(
            [1, queryHeads, chunk, headDim], key: MLXRandom.key(9_302)
        ).asType(dtype)
        let queries = concatenated([queries0, queries1], axis: 0)

        let (k0, v0) = codedKV(0 ..< chunk, tag: 0, dtype: dtype)
        // Two wildly different batchmate contents at the SAME positions.
        let (kA, vA) = codedKV(0 ..< chunk, tag: 5, dtype: dtype)
        let (kB, vB) = codedKV(0 ..< chunk, tag: 41, dtype: dtype)

        func run(batchmateKeys: MLXArray, batchmateValues: MLXArray) throws -> MLXArray {
            var states: [[CBv2SequenceKV?]] = []
            var rows: [PagedSequenceKV] = []
            for tag in 0 ..< 2 {
                let made = try makeRow(
                    backend: backend, layer: layer, tag: tag, priming: 0,
                    maxLength: chunk, dtype: dtype)
                states.append(made.state)
                rows.append(made.row)
            }
            defer { for s in states { backend.release(s) } }
            let cache = backend.makeLayerCaches()[0]
            cache.setRows(rows)
            let out = cache.updateAndAttend(
                queries: queries,
                keys: concatenated(
                    [k0.expandedDimensions(axis: 0), batchmateKeys.expandedDimensions(axis: 0)],
                    axis: 0),
                values: concatenated(
                    [
                        v0.expandedDimensions(axis: 0),
                        batchmateValues.expandedDimensions(axis: 0),
                    ], axis: 0),
                scale: scale, sinks: nil)
            eval(out)
            return out
        }

        let withA = try run(batchmateKeys: kA, batchmateValues: vA)
        let withB = try run(batchmateKeys: kB, batchmateValues: vB)

        assertIdentical(
            withA[0 ..< 1], withB[0 ..< 1],
            "row 0 with two completely different batchmates")

        // Premise: the batchmate swap really did change that row's own
        // answer, so the invariance above is a property of ROW SEPARATION
        // and not of an attention that ignores its keys.
        let batchmateDelta =
            abs(withA[1 ..< 2].asType(.float32) - withB[1 ..< 2].asType(.float32))
            .max().item(Float.self)
        #expect(
            batchmateDelta > 1e-3,
            "premise: swapping row 1's content must change row 1's own output (max |delta| = \(batchmateDelta))")
    }

    // MARK: - WS-2.2: vision span masks

    /// The chunk view a paged prefill of `chunk` tokens from offset 0
    /// attends: the chunk itself, nothing before it.
    private func spanReference(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        window: Int?, context: CBv2SpanChunkContext
    ) -> MLXArray {
        // The CONTIGUOUS backend's mask builder, applied to the paged K/V.
        // If paged's absolute-coordinate overlay disagrees with the
        // relative-coordinate one by a single position, this diverges.
        let mask = CBv2AttentionV1.spanChunkMask(
            L: queries.dim(2), kL: keys.dim(2), window: window, context: context)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask))
    }

    /// A span-masked chunk on paged must equal the same chunk served by the
    /// CONTIGUOUS backend — same layer kind, same K/V, same bound context.
    /// This is the end-to-end statement of "vision serving works on paged".
    @Test func spanMaskedChunkMatchesContiguousBackend() throws {
        for window in [nil, 128] as [Int?] {
            let layer = kind(window: window)
            let chunk = 96
            let dtype = DType.float32
            let context = CBv2SpanChunkContext(
                chunkEnd: chunk,
                blocks: [
                    CBv2ImageSpan(tokenOffset: 8, length: 20),
                    CBv2ImageSpan(tokenOffset: 55, length: 16),
                ])

            let (k, v) = codedKV(0 ..< chunk, tag: 3, dtype: dtype)
            let keys = k.expandedDimensions(axis: 0)
            let values = v.expandedDimensions(axis: 0)
            let queries = MLXRandom.normal(
                [1, queryHeads, chunk, headDim], key: MLXRandom.key(9_401)
            ).asType(dtype)

            // Paged.
            let backend = try PagedKVBackend(
                layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
            let state = try backend.makeSequenceState(
                layerKinds: [layer], promptLength: chunk, maxLength: chunk)
            defer { backend.release(state) }
            let paged = backend.makeLayerCaches()[0]
            paged.setRows([state[0]!])
            paged.bindSpanContext(context)
            let got = paged.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
            paged.bindSpanContext(nil)

            // Contiguous, same everything.
            let contiguousBackend = CBv2ContiguousKVBackend(
                config: CBv2ContiguousBackendConfig(bytesCapacity: 64 << 20, kvDType: dtype))
            let contiguousState = try contiguousBackend.makeSequenceState(
                layerKinds: [layer], promptLength: chunk, maxLength: chunk)
            defer { contiguousBackend.release(contiguousState) }
            let contiguous = CBv2LayerCache(
                layerIndex: 0, kind: layer, rows: [contiguousState[0]!])
            contiguous.bindSpanContext(context)
            let want = contiguous.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
            contiguous.bindSpanContext(nil)

            assertClose(got, want, "paged vs contiguous span chunk (window \(String(describing: window)))")

            // And term for term against the contiguous mask builder itself.
            assertClose(
                got,
                spanReference(
                    queries: queries, keys: keys, values: values,
                    window: window, context: context),
                "paged span chunk vs CBv2AttentionV1.spanChunkMask (window \(String(describing: window)))"
            )
        }
    }

    /// The overlay must actually DO something: a bound span has to change
    /// the answer versus the same chunk served causally. Without this, a
    /// `bindSpanContext` that stored the context and never read it would
    /// pass every comparison above that used a reference built the same
    /// wrong way.
    @Test func boundSpanChangesTheResultAndUnbindingRestoresCausal() throws {
        let layer = kind(window: nil)
        let chunk = 96
        let dtype = DType.float32
        let context = CBv2SpanChunkContext(
            chunkEnd: chunk, blocks: [CBv2ImageSpan(tokenOffset: 10, length: 40)])

        let (k, v) = codedKV(0 ..< chunk, tag: 7, dtype: dtype)
        let keys = k.expandedDimensions(axis: 0)
        let values = v.expandedDimensions(axis: 0)
        let queries = MLXRandom.normal(
            [1, queryHeads, chunk, headDim], key: MLXRandom.key(9_501)
        ).asType(dtype)

        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))

        func attend(binding: CBv2SpanChunkContext?) throws -> MLXArray {
            let state = try backend.makeSequenceState(
                layerKinds: [layer], promptLength: chunk, maxLength: chunk)
            defer { backend.release(state) }
            let cache = backend.makeLayerCaches()[0]
            cache.setRows([state[0]!])
            cache.bindSpanContext(binding)
            let out = cache.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
            eval(out)
            #expect(
                (cache.boundSpanContext == nil) == (binding == nil),
                "the cache must hold exactly what was bound")
            cache.bindSpanContext(nil)
            #expect(cache.boundSpanContext == nil, "unbinding must clear the context")
            return out
        }

        let spanned = try attend(binding: context)
        let causal = try attend(binding: nil)
        let delta = abs(spanned.asType(.float32) - causal.asType(.float32)).max().item(Float.self)
        #expect(
            delta > 1e-3,
            "a bound span must change the result versus plain causal (max |delta| = \(delta))")

        // Rebinding nil returns to exactly the causal answer — the context is
        // per-chunk state, not a latch.
        let causalAgain = try attend(binding: nil)
        assertIdentical(causal, causalAgain, "unbound chunk after a span chunk")
    }

    /// Span chunks must stay UNBLOCKED. The block loop slices keys up to the
    /// LATEST query's own position, which is precisely the causal bound a
    /// bidirectional span exists to escape: here the image block straddles
    /// the 128-token block boundary, so a query at 100 has to attend a key
    /// at 150 that a blocked slice would not even contain.
    @Test func spanChunkStaysUnblockedAcrossTheQueryBlockBoundary() throws {
        let blockSize = CBv2AttentionV1.queryBlockSize
        try #require(blockSize > 0, "kill switch set: blocking is off, nothing to prove")
        let layer = kind(window: nil)
        let chunk = blockSize + 64
        let dtype = DType.float32
        #expect(CBv2AttentionV1.shouldBlockQueries(chunk), "premise: this chunk WOULD be blocked")

        // A block that starts before and ends after the query-block seam.
        let spanStart = blockSize - 28
        let spanLength = 56
        #expect(spanStart < blockSize && spanStart + spanLength > blockSize)
        let context = CBv2SpanChunkContext(
            chunkEnd: chunk,
            blocks: [CBv2ImageSpan(tokenOffset: spanStart, length: spanLength)])

        let (k, v) = codedKV(0 ..< chunk, tag: 11, dtype: dtype)
        let keys = k.expandedDimensions(axis: 0)
        let values = v.expandedDimensions(axis: 0)
        let queries = MLXRandom.normal(
            [1, queryHeads, chunk, headDim], key: MLXRandom.key(9_601)
        ).asType(dtype)

        let backend = try PagedKVBackend(
            layerKinds: [layer], config: config(maxPrefillChunk: chunk, dtype: dtype))
        let state = try backend.makeSequenceState(
            layerKinds: [layer], promptLength: chunk, maxLength: chunk)
        defer { backend.release(state) }
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([state[0]!])
        cache.bindSpanContext(context)
        let got = cache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        cache.bindSpanContext(nil)

        assertClose(
            got,
            spanReference(
                queries: queries, keys: keys, values: values, window: nil, context: context),
            "span chunk straddling the query-block seam")
    }

    // MARK: - Capability gates

    /// Both flags, read the way the bank reads them.
    @Test func pagedBankAffirmsBothWS2Capabilities() throws {
        let kinds = [kind(window: nil), kind(window: 128)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 64, dtype: .float16))
        let caches = backend.makeLayerCaches()
        for cache in caches {
            #expect(cache.keepsRowsIndependentWhenPacked)
            #expect(cache.honorsSpanMaskContexts)
        }
        let bank = CBv2LayerCacheBank(caches: caches)
        #expect(
            bank.supportsPackedPrefill,
            "the paged cache now implements packed prefill, so the bank must vouch")
        #expect(
            bank.supportsMultimodalSpans,
            "the paged cache now applies bound span contexts, so the bank must vouch")
    }

    /// The provider's slot policy routes VLM traffic to paged by reading the
    /// TYPE-level constants, because it must decide before any pool exists.
    /// That is only safe while the constants say what a real bank of real
    /// caches says — a static that drifts from its instances would route
    /// vision at a backend that no longer honours it, which is the exact
    /// failure the affirmative-capability design exists to prevent.
    ///
    /// Cross-repo consumer: `EngineV2KVBackendPolicy.applySlotVetoes` in
    /// provider-swift takes `pagedHonorsSpanMasks:` and
    /// `EngineV2SlotFactory` feeds it
    /// `PagedLayerCache.honorsSpanMaskContextsByConstruction`.
    @Test func typeLevelClaimsMatchARealPagedBank() throws {
        let kinds = [kind(window: nil), kind(window: 128), kind(window: 128)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 64, dtype: .float16))
        let caches = backend.makeLayerCaches()
        let bank = CBv2LayerCacheBank(caches: caches)

        #expect(
            bank.supportsMultimodalSpans
                == PagedLayerCache.honorsSpanMaskContextsByConstruction,
            "the constant the provider routes on must equal what a real paged bank answers")
        #expect(
            bank.supportsPackedPrefill
                == PagedLayerCache.keepsRowsIndependentWhenPackedByConstruction,
            "same for packed prefill")

        // Every instance derives from the constant, so no single layer can
        // disagree with the type — including a windowed one, a full one, and
        // a second windowed one sharing the first's group.
        for cache in caches {
            #expect(
                cache.honorsSpanMaskContexts
                    == PagedLayerCache.honorsSpanMaskContextsByConstruction)
            #expect(
                cache.keepsRowsIndependentWhenPacked
                    == PagedLayerCache.keepsRowsIndependentWhenPackedByConstruction)
        }
    }

    /// The gates read the CLAIM, not the concrete type. A cache that is
    /// neither `CBv2LayerCache` nor `PagedLayerCache` opens the gates by
    /// answering true, and closes them by answering false — an `is` cast on
    /// either shipping class could not tell these two apart.
    @Test func bankGatesReadTheClaimNotTheConcreteType() {
        let layer = kind(window: nil)
        let affirming = CBv2ClaimingLayerCache(layerIndex: 0, kind: layer, claim: true)
        let refusing = CBv2ClaimingLayerCache(layerIndex: 0, kind: layer, claim: false)

        let open = CBv2LayerCacheBank(caches: [affirming])
        #expect(open.supportsPackedPrefill, "an affirmative claim from an unknown class opens it")
        #expect(open.supportsMultimodalSpans)

        let closed = CBv2LayerCacheBank(caches: [refusing])
        #expect(
            !closed.supportsPackedPrefill,
            "the same class answering false must close the gate — so the gate is the value, not the conformance and not the type")
        #expect(!closed.supportsMultimodalSpans)

        // Mixed with a real paged cache: one refusal closes the whole bank.
        let mixed = CBv2LayerCacheBank(caches: [affirming, refusing])
        #expect(!mixed.supportsPackedPrefill)
        #expect(!mixed.supportsMultimodalSpans)
    }
}

// MARK: - A capability-claiming cache of an unrelated class

/// Neither shipping cache class. Exists solely so the bank's gates can be
/// shown to read `keepsRowsIndependentWhenPacked` / `honorsSpanMaskContexts`
/// rather than `cache is CBv2LayerCache`. Attention is never invoked.
final class CBv2ClaimingLayerCache: CBv2AttendingLayerCache {
    let layerIndex: Int
    let kind: CBv2LayerKind
    let claim: Bool
    private(set) var rows: [CBv2SequenceKV] = []
    private(set) var spanContext: CBv2SpanChunkContext?

    init(layerIndex: Int, kind: CBv2LayerKind, claim: Bool) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.claim = claim
    }

    func setRows(_ rows: [CBv2SequenceKV]) { self.rows = rows }
    var positionOffsets: MLXArray { MLXArray([] as [Int32]) }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        fatalError("CBv2ClaimingLayerCache is a capability fixture, not an attention path")
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        fatalError("CBv2ClaimingLayerCache is a capability fixture, not an attention path")
    }
}

extension CBv2ClaimingLayerCache: CBv2PackedPrefillCapableCache {
    var keepsRowsIndependentWhenPacked: Bool { claim }
}

extension CBv2ClaimingLayerCache: CBv2MultimodalSpanCapableCache {
    func bindSpanContext(_ context: CBv2SpanChunkContext?) { spanContext = context }
    var honorsSpanMaskContexts: Bool { claim }
}
