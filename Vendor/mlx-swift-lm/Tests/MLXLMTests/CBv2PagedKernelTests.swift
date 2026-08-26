// CBv2PagedKernelTests.swift
//
// WS-C kernel correctness: paged-attention decode kernel parity vs the
// composed fp32 reference (rel-err <= 1e-2 fp16), batch-composition
// invariance (bitwise), prefill fallback parity, KV-shared borrowing, and
// a 200-step greedy decode token-match. No model weights required.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedKernel", .serialized)
struct CBv2PagedKernelTests {

    // MARK: - Config helper

    /// A paged config whose `maxPrefillChunk` the pool will actually accept
    /// for `kind`.
    ///
    /// The windowed ring is `ceil(max(window + maxSpeculativeSpan,
    /// maxPrefillChunk) / pageSize)` pages, and `PagedKVPool` REFUSES a config
    /// whose `maxPrefillChunk` exceeds the whole ring — a chunk that laps the
    /// ring would put two of its own tokens in one physical slot inside a
    /// single dispatch. In practice the `max` means the ring always covers the
    /// chunk, so this clamp is belt-and-braces; it stays because the shape
    /// matrix is declarative and a future sizing change should not silently
    /// turn a fixture into a `backendIneligible` throw.
    ///
    /// Full-attention layers have no ring and are unclamped.
    static func pagedConfig(
        for kind: CBv2LayerKind, capacityBytes: Int, desiredPrefillChunk: Int = 64,
        nominalMaxSequenceLength: Int = 4096
    ) -> PagedKVPoolConfig {
        var config = PagedKVPoolConfig(
            capacityBytes: capacityBytes,
            maxPrefillChunk: desiredPrefillChunk,
            nominalMaxSequenceLength: nominalMaxSequenceLength)
        if case .slidingWindow(let window) = kind.attention {
            let ringTokens =
                PagedKVPool.ringPageCount(window: window, config: config) * config.pageSize
            config.maxPrefillChunk = min(desiredPrefillChunk, ringTokens)
        }
        return config
    }

    /// Largest chunk that may be handed to `PagedSequenceKV.write` DIRECTLY —
    /// i.e. bypassing `PagedLayerCache`, as the fixtures do when they seed a
    /// row's history.
    ///
    /// `config.maxPrefillChunk`, same as any other writer. It was NOT always:
    /// a direct write used to leave `retainedCount == min(written, window - 1
    /// + n)`, so the following `attendableViews()` asked the ring for
    /// `window - 1 + n` and `gatherRange` aborted the process once that span
    /// exceeded the ring — bounding a direct writer at `ringTokens - window +
    /// 1`, which is 17 tokens at gemma-4's geometry against a 512-token chunk.
    ///
    /// WS-1.2's row half removed that. `PagedSequenceKV.update` now gathers a
    /// chunk's window history BEFORE writing the chunk, exactly as
    /// `PagedLayerCache.prefillKV` does, so `retainedCount` is
    /// `min(written, window)` in every phase and the ring only has to cover
    /// one whole chunk — which `PagedKVPool.ringPageCount`'s ROW bound
    /// guarantees by construction.
    ///
    /// Kept as a named function rather than inlined so the distinction stays
    /// greppable: it is the thing that used to be true, and the two callers it
    /// stands for (this harness and `PagedDecodeProfiler`) are the paged
    /// backend's only direct writers.
    static func rowSafeWriteChunk(for kind: CBv2LayerKind, desired: Int = 64) -> Int {
        pagedConfig(for: kind, capacityBytes: 1 << 20, desiredPrefillChunk: desired)
            .maxPrefillChunk
    }

    // MARK: - Fixtures

    /// One attention layer + its paged plumbing plus an independent
    /// contiguous "mirror" of everything written, used to compute
    /// references without touching the pool.
    final class Fixture {
        let kind: CBv2LayerKind
        let backend: PagedKVBackend
        let cache: PagedLayerCache
        var states: [[CBv2SequenceKV?]] = []
        var rows: [PagedSequenceKV] = []
        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        /// Largest chunk `row.write` accepts — see `pagedConfig(for:...)`.
        let writeChunk: Int

        init(
            kind: CBv2LayerKind, maxPrefillChunk: Int = 64,
            attentionSoftcap: Float? = nil
        ) throws {
            self.kind = kind
            let config = CBv2PagedKernelTests.pagedConfig(
                for: kind, capacityBytes: 64 << 20,
                desiredPrefillChunk: maxPrefillChunk)
            self.writeChunk = CBv2PagedKernelTests.rowSafeWriteChunk(
                for: kind, desired: config.maxPrefillChunk)
            self.backend = try PagedKVBackend(layerKinds: [kind], config: config)
            self.cache = backend.makeLayerCaches(attentionSoftcap: attentionSoftcap)[0]
        }

        deinit {
            states.forEach { backend.release($0) }
        }

        @discardableResult
        func addRow(tokens: Int, maxLength: Int = 2048) throws -> Int {
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: tokens, maxLength: maxLength)
            let row = state[0] as! PagedSequenceKV
            states.append(state)
            rows.append(row)
            var k = MLXArray.zeros([kind.kvHeads, 0, kind.headDim], dtype: .float16)
            var v = k
            var remaining = tokens
            while remaining > 0 {
                let n = min(remaining, writeChunk)
                let ck = MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16)
                let cv = MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16)
                row.write(keys: ck, values: cv)
                k = concatenated([k, ck], axis: 1)
                v = concatenated([v, cv], axis: 1)
                remaining -= n
            }
            mirrorK.append(k)
            mirrorV.append(v)
            cache.setRows(rows)
            return rows.count - 1
        }

        /// Reference decode for row `i` given this step's (q, k, v) —
        /// window clamping recomputed independently from the mirror.
        func referenceDecode(
            rowIndex i: Int, q: MLXArray, newK: MLXArray, newV: MLXArray,
            sinks: MLXArray?, scale: Float, softcap: Float? = nil
        ) -> MLXArray {
            mirrorK[i] = concatenated([mirrorK[i], newK], axis: 1)
            mirrorV[i] = concatenated([mirrorV[i], newV], axis: 1)
            let t = mirrorK[i].dim(1)
            var start = 0
            if case .slidingWindow(let w) = kind.attention {
                start = max(0, t - w)
            }
            let k = mirrorK[i][0..., start ..< t, 0...].expandedDimensions(axis: 0)
            let v = mirrorV[i][0..., start ..< t, 0...].expandedDimensions(axis: 0)
            return PagedAttentionReference.composedAttention(
                queries: q.expandedDimensions(axis: 0), keys: k, values: v,
                scale: scale, sinks: sinks, softcap: softcap)
        }
    }

    /// The backends-to-each-other verdict, as a predicate: "do these two
    /// arrays agree within `rtol / atol`". Factored out of `assertClose` so
    /// the calibration test can interrogate the SAME bar it asserts, rather
    /// than a re-derivation that might drift from it.
    private static func arraysAgree(
        _ a: MLXArray, _ b: MLXArray, rtol: Float = 1e-2, atol: Float = 2e-3
    ) -> Bool {
        allClose(
            a.asType(.float32), b.asType(.float32), rtol: Double(rtol), atol: Double(atol)
        ).item(Bool.self)
    }

    private func assertClose(
        _ got: MLXArray, _ want: MLXArray, rtol: Float = 1e-2, atol: Float = 2e-3,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        if !Self.arraysAgree(got, want, rtol: rtol, atol: atol) {
            let diff = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
            Issue.record(
                "arrays differ, max abs err \(diff)", sourceLocation: sourceLocation)
        }
    }

    // MARK: - Kernel parity vs composed reference

    @Test(arguments: [
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: true),
        (headDim: 64, kvHeads: 2, queryHeads: 16, window: Int?(20), sinks: true),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?(33), sinks: false),
        (headDim: 256, kvHeads: 2, queryHeads: 4, window: Int?.none, sinks: false),
        (headDim: 512, kvHeads: 2, queryHeads: 4, window: Int?.none, sinks: false),
        // Gemma-4-26B global-layer shape (d512, GQA 8) — exercises the
        // head split (HPT=2, 4 threadgroups per kv head).
        (headDim: 512, kvHeads: 2, queryHeads: 16, window: Int?.none, sinks: false),
        (headDim: 512, kvHeads: 1, queryHeads: 8, window: Int?(24), sinks: false),
    ])
    func decodeKernelParity(
        _ shape: (headDim: Int, kvHeads: Int, queryHeads: Int, window: Int?, sinks: Bool)
    ) throws {
        MLXRandom.seed(42)
        let attention: CBv2LayerKind.Attention =
            shape.window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, hasSinks: shape.sinks, headDim: shape.headDim,
            kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)
        let fixture = try Fixture(kind: kind)

        // Mixed row lengths, including page-boundary and sub-page cases.
        for tokens in [4, 16, 33, 100] {
            try fixture.addRow(tokens: tokens)
        }
        let b = fixture.rows.count
        let scale = Float(1.0 / Double(shape.headDim).squareRoot())
        let sinks: MLXArray? =
            shape.sinks
            ? MLXRandom.normal([shape.queryHeads], dtype: .float32) : nil

        // A few decode steps so positions advance past page boundaries.
        for _ in 0 ..< 3 {
            let q = MLXRandom.normal(
                [b, shape.queryHeads, 1, shape.headDim], dtype: .float16)
            let k = MLXRandom.normal(
                [b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let v = MLXRandom.normal(
                [b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let out = fixture.cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            #expect(out.shape == [b, shape.queryHeads, 1, shape.headDim])
            for i in 0 ..< b {
                let ref = fixture.referenceDecode(
                    rowIndex: i, q: q[i], newK: k[i], newV: v[i],
                    sinks: sinks, scale: scale)
                assertClose(out[i].expandedDimensions(axis: 0), ref)
            }
        }
    }

    @Test func decodeKernelSoftcapParity() throws {
        MLXRandom.seed(7)
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind, attentionSoftcap: 30.0)
        try fixture.addRow(tokens: 50)
        let scale: Float = 0.125

        let q = MLXRandom.normal([1, 4, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        let ref = fixture.referenceDecode(
            rowIndex: 0, q: q[0], newK: k[0], newV: v[0],
            sinks: nil, scale: scale, softcap: 30.0)
        assertClose(out[0].expandedDimensions(axis: 0), ref)
    }

    // MARK: - Batch-composition invariance

    @Test func decodeBatchCompositionInvariance() throws {
        MLXRandom.seed(11)
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 8)
        let scale: Float = 0.125

        // Two fixtures: the probe row solo, and the SAME content batched
        // with two batchmates. Identical content => bitwise-identical
        // outputs, or the backend leaks batch composition. The 600-token
        // batchmate spans multiple flash-decoding partitions, so the
        // batched dispatch launches MORE partition threadgroups than the
        // solo one — the probe row's math must not notice.
        let solo = try Fixture(kind: kind)
        let batched = try Fixture(kind: kind)

        MLXRandom.seed(100)
        try solo.addRow(tokens: 37)
        MLXRandom.seed(100)
        try batched.addRow(tokens: 37)
        try batched.addRow(tokens: 600)
        try batched.addRow(tokens: 90)

        MLXRandom.seed(200)
        let q = MLXRandom.normal([1, 8, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let qB = concatenated(
            [q, MLXRandom.normal([2, 8, 1, 64], dtype: .float16)], axis: 0)
        let kB = concatenated(
            [k, MLXRandom.normal([2, 2, 1, 64], dtype: .float16)], axis: 0)
        let vB = concatenated(
            [v, MLXRandom.normal([2, 2, 1, 64], dtype: .float16)], axis: 0)

        let outSolo = solo.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        let outBatched = batched.cache.updateAndAttend(
            queries: qB, keys: kB, values: vB, scale: scale, sinks: nil)

        #expect(
            arrayEqual(outSolo[0], outBatched[0]).item(Bool.self),
            "row output depends on batchmates — invariance violation")
    }

    // MARK: - Prefill fallback

    @Test(arguments: [Int?.none, Int?(24)])
    func prefillChunkParity(window: Int?) throws {
        MLXRandom.seed(3)
        let attention: CBv2LayerKind.Attention =
            window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind, maxPrefillChunk: 32)
        try fixture.addRow(tokens: 0)
        let row = fixture.rows[0]
        let scale: Float = 0.125

        var mirrorK = MLXArray.zeros([1, 2, 0, 64], dtype: .float16)
        var mirrorV = mirrorK
        for chunk in [9, 32, 16] {
            let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
            let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let out = fixture.cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: nil)
            mirrorK = concatenated([mirrorK, k], axis: 2)
            mirrorV = concatenated([mirrorV, v], axis: 2)

            // Reference: full mirror + explicit causal/window bool mask.
            let t = mirrorK.dim(2)
            let qpos = MLXArray(Int32(t - chunk) ..< Int32(t)).expandedDimensions(axis: 1)
            let kpos = MLXArray(Int32(0) ..< Int32(t)).expandedDimensions(axis: 0)
            var mask = kpos .<= qpos
            if let window {
                mask = mask & (kpos .> (qpos - Int32(window)))
            }
            let ref = PagedAttentionReference.composedAttention(
                queries: q, keys: mirrorK, values: mirrorV, scale: scale,
                boolMask: mask, sinks: nil)
            assertClose(out, ref)
            #expect(row.absoluteOffset == t)
        }

        // A decode step right after chunked prefill must agree too
        // (prefill -> decode transition over the same ring).
        let q = MLXRandom.normal([1, 4, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        mirrorK = concatenated([mirrorK, k], axis: 2)
        mirrorV = concatenated([mirrorV, v], axis: 2)
        let t = mirrorK.dim(2)
        var start = 0
        if let window { start = max(0, t - window) }
        let ref = PagedAttentionReference.composedAttention(
            queries: q,
            keys: mirrorK[0..., 0..., start ..< t, 0...],
            values: mirrorV[0..., 0..., start ..< t, 0...],
            scale: scale, sinks: nil)
        assertClose(out, ref)
    }

    @Test func prefillWithSinksParity() throws {
        MLXRandom.seed(13)
        let kind = CBv2LayerKind(
            attention: .full, hasSinks: true, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 0)
        let sinks = MLXRandom.normal([4], dtype: .float32)
        let scale: Float = 0.125

        let chunk = 12
        let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: sinks)

        let qpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 0)
        let ref = PagedAttentionReference.composedAttention(
            queries: q, keys: k, values: v, scale: scale,
            boolMask: kpos .<= qpos, sinks: sinks)
        assertClose(out, ref)
    }

    // MARK: - KV-shared borrowing (Gemma-style)

    @Test func borrowingMatchesSourceStorage() throws {
        MLXRandom.seed(21)
        let owner = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 8)
        var shared = owner
        shared.sharesKVWithLayer = 0
        let kinds = [owner, shared]
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(capacityBytes: 32 << 20))
        let caches = backend.makeLayerCaches()
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        let row = state[0] as! PagedSequenceKV
        caches[0].setRows([row])
        let scale: Float = 0.125

        var mirrorK = MLXArray.zeros([2, 0, 64], dtype: .float16)
        var mirrorV = mirrorK
        // Prefill through the owner, then decode; borrow at each decode.
        let pk = MLXRandom.normal([1, 2, 30, 64], dtype: .float16)
        let pv = MLXRandom.normal([1, 2, 30, 64], dtype: .float16)
        _ = caches[0].updateAndAttend(
            queries: MLXRandom.normal([1, 8, 30, 64], dtype: .float16),
            keys: pk, values: pv, scale: scale, sinks: nil)
        mirrorK = concatenated([mirrorK, pk.squeezed(axis: 0)], axis: 1)
        mirrorV = concatenated([mirrorV, pv.squeezed(axis: 0)], axis: 1)

        for _ in 0 ..< 2 {
            let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
            _ = caches[0].updateAndAttend(
                queries: MLXRandom.normal([1, 8, 1, 64], dtype: .float16),
                keys: k, values: v, scale: scale, sinks: nil)
            mirrorK = concatenated([mirrorK, k.squeezed(axis: 0)], axis: 1)
            mirrorV = concatenated([mirrorV, v.squeezed(axis: 0)], axis: 1)

            // The shared layer attends the owner's K/V with its own queries.
            let qShared = MLXRandom.normal([1, 8, 1, 64], dtype: .float16)
            let out = caches[1].attendBorrowing(
                source: caches[0], queries: qShared, scale: scale, sinks: nil)
            let ref = PagedAttentionReference.composedAttention(
                queries: qShared,
                keys: mirrorK.expandedDimensions(axis: 0),
                values: mirrorV.expandedDimensions(axis: 0),
                scale: scale, sinks: nil)
            assertClose(out, ref)
        }
        backend.release(state)
    }

    // MARK: - Position offsets

    @Test func positionOffsetsMatchAbsolutePositions() throws {
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 12)
        try fixture.addRow(tokens: 40)
        let offsets = fixture.cache.positionOffsets
        #expect(offsets.shape == [2])
        #expect(offsets[0].item(Int32.self) == 12)
        #expect(offsets[1].item(Int32.self) == 40)
    }

    // MARK: - End-to-end greedy decode token match

    /// 200 greedy decode steps on a tiny random attention "model": the
    /// paged kernel path and the composed contiguous reference each feed
    /// their own argmax back. Correctness bar: >= 99.5% token match.
    @Test func greedyDecodeTokenMatch200Steps() throws {
        MLXRandom.seed(1234)
        let vocab = 97
        let headDim = 64
        let kvHeads = 2
        let queryHeads = 4

        let embedQ = MLXRandom.normal([vocab, queryHeads * headDim], dtype: .float16)
        let embedK = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let embedV = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let unembed = MLXRandom.normal([queryHeads * headDim, vocab], dtype: .float16)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        let kind = CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 0, maxLength: 512)

        func project(_ token: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
            let q = embedQ[token].reshaped([1, queryHeads, 1, headDim])
            let k = embedK[token].reshaped([1, kvHeads, 1, headDim])
            let v = embedV[token].reshaped([1, kvHeads, 1, headDim])
            return (q, k, v)
        }
        func logits(_ attnOut: MLXArray) -> MLXArray {
            matmul(attnOut.reshaped([1, queryHeads * headDim]).asType(.float32),
                   unembed.asType(.float32))
        }

        var pagedToken = 1
        var refToken = 1
        var mirrorK = MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16)
        var mirrorV = mirrorK
        var matches = 0
        let steps = 200

        for _ in 0 ..< steps {
            // Paged path.
            let p = project(pagedToken)
            let outPaged = fixture.cache.updateAndAttend(
                queries: p.q, keys: p.k, values: p.v, scale: scale, sinks: nil)
            let nextPaged = argMax(logits(outPaged), axis: -1).item(Int32.self)

            // Composed contiguous reference path.
            let r = project(refToken)
            mirrorK = concatenated([mirrorK, r.k], axis: 2)
            mirrorV = concatenated([mirrorV, r.v], axis: 2)
            let outRef = PagedAttentionReference.composedAttention(
                queries: r.q, keys: mirrorK, values: mirrorV, scale: scale, sinks: nil)
            let nextRef = argMax(logits(outRef), axis: -1).item(Int32.self)

            if nextPaged == nextRef { matches += 1 }
            pagedToken = Int(nextPaged)
            refToken = Int(nextRef)
        }
        #expect(
            Double(matches) >= 0.995 * Double(steps),
            "greedy token match \(matches)/\(steps) below 99.5%")
    }

    // MARK: - Cross-backend differential: paged vs the CONTIGUOUS engine
    //
    // Everything above this mark compares the paged pool against an in-file
    // fp32 recomputation over a MIRROR of the same writes. That is a strong
    // oracle for the ATTENTION math and a weak one for STORAGE: a bug in the
    // shared write path — a mis-sized ring, a slipped page bound, a pad entry
    // that lands somewhere live — corrupts the pool while the mirror stays
    // clean only if the mirror is an independent STORAGE ENGINE. It is not; it
    // is a Swift array of the tensors the test itself generated, so a write
    // that never reached the slab and a read that never happened cancel out.
    //
    // These tests replace the mirror with `CBv2ContiguousKVBackend` +
    // `CBv2LayerCache`: a second, independently implemented store (contiguous
    // per-row buffers read by MLXFast SDPA) fed byte-identical inputs. The two
    // engines are compared against EACH OTHER, so agreement requires both to
    // have stored, evicted and re-read the same values.

    private enum StorageArm {
        case paged
        case contiguous
    }

    /// One layer of one storage engine, driven through the shared
    /// `CBv2AttendingLayerCache` surface. History is appended in identical
    /// chunk sizes on both arms so they observe identical `lastUpdateTokens`
    /// trajectories (which is what governs a windowed row's retained span).
    private final class Arm {
        let cache: CBv2AttendingLayerCache
        private let release: () -> Void

        init(
            _ arm: StorageArm, kind: CBv2LayerKind, histories: [[MLXArray]],
            maxLength: Int
        ) throws {
            let kinds = [kind]
            var rows: [CBv2SequenceKV] = []
            switch arm {
            case .paged:
                let backend = try PagedKVBackend(
                    layerKinds: kinds,
                    config: CBv2PagedKernelTests.pagedConfig(
                        for: kind, capacityBytes: 128 << 20))
                cache = backend.makeLayerCaches()[0]
                var states: [[CBv2SequenceKV?]] = []
                for chunks in histories {
                    let state = try backend.makeSequenceState(
                        layerKinds: kinds, promptLength: 0, maxLength: maxLength)
                    let row = state[0] as! PagedSequenceKV
                    for pair in stride(from: 0, to: chunks.count, by: 2) {
                        row.write(keys: chunks[pair], values: chunks[pair + 1])
                    }
                    states.append(state)
                    rows.append(row)
                }
                release = { states.forEach { backend.release($0) } }
            case .contiguous:
                let backend = CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 1 << 29))
                cache = CBv2LayerCache(layerIndex: 0, kind: kind)
                var states: [[CBv2SequenceKV?]] = []
                for chunks in histories {
                    let state = try backend.makeSequenceState(
                        layerKinds: kinds, promptLength: 0, maxLength: maxLength)
                    let row = state[0]!
                    for pair in stride(from: 0, to: chunks.count, by: 2) {
                        _ = row.update(
                            keys: chunks[pair].expandedDimensions(axis: 0),
                            values: chunks[pair + 1].expandedDimensions(axis: 0))
                    }
                    states.append(state)
                    rows.append(row)
                }
                release = { states.forEach { backend.release($0) } }
            }
            cache.setRows(rows)
        }

        deinit { release() }

        func step(
            queries: MLXArray, keys: MLXArray, values: MLXArray,
            scale: Float, sinks: MLXArray?
        ) -> MLXArray {
            cache.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: sinks)
        }
    }

    /// Tensor source for the cross-backend tests, deterministic in its SEED
    /// ALONE.
    ///
    /// `MLXRandom.seed` sets a PROCESS-GLOBAL key and Swift Testing runs
    /// suites in parallel, so another suite drawing from MLXRandom between
    /// two of this suite's draws changes the tensors this suite receives.
    /// Under `--filter CBv2PagedKernel` nothing else runs and the global seed
    /// looks perfectly reproducible; in a full-suite run it is not.
    ///
    /// For the differential that is merely untidy — both arms are handed
    /// whatever was drawn, so the comparison stands either way. For
    /// `differentialOracleRejectsPlantedFaults` it is fatal: whether the
    /// oldest key carries any softmax mass is a property OF THE DRAW, so a
    /// perturbed draw silently converts the oldest-key probe from
    /// discriminating into vacuous. It was caught exactly that way — an
    /// intermittent 6.9e-08 on `gemma4-sliding history 200` during a
    /// full-suite run, against 3.9e-1 for the identical probe under the
    /// filter. A calibration that measures a different fixture each run
    /// calibrates nothing.
    ///
    /// A split-key chain removes the shared mutable state entirely.
    struct KeyedRandom {
        private var key: MLXArray

        init(seed: UInt64) { key = MLXRandom.key(seed) }

        mutating func normal(_ shape: [Int], dtype: DType = .float16) -> MLXArray {
            let (next, draw) = MLXRandom.split(key: key)
            key = next
            return MLXRandom.normal(shape, dtype: dtype, key: draw)
        }
    }

    /// `[kvHeads, n, headDim]` history chunks per row, generated ONCE and
    /// handed to both arms so the two engines store byte-identical bytes AND
    /// observe identical `lastUpdateTokens` trajectories.
    ///
    /// Chunk width is whatever the paged pool will accept for `kind` (see
    /// `pagedConfig(for:...)`); the contiguous arm has no such bound but must
    /// use the same widths or the two rows' retained spans diverge for reasons
    /// that have nothing to do with storage.
    private func makeHistories(
        kind: CBv2LayerKind, lengths: [Int], rng: inout KeyedRandom
    ) -> [[MLXArray]] {
        let chunk = CBv2PagedKernelTests.rowSafeWriteChunk(for: kind)
        return lengths.map { tokens in
            var chunks: [MLXArray] = []
            var remaining = tokens
            while remaining > 0 {
                let n = min(remaining, chunk)
                chunks.append(rng.normal([kind.kvHeads, n, kind.headDim]))
                chunks.append(rng.normal([kind.kvHeads, n, kind.headDim]))
                remaining -= n
            }
            return chunks
        }
    }

    // MARK: - The third point: an fp32 reference over the same stored values
    //
    // Comparing the two storage engines TO EACH OTHER answers "do they
    // agree", never "are they right". A systematic error both arms share is
    // invisible to `assertClose(paged, contiguous)`: at `rtol 1e-2 / atol
    // 2e-3` a paged arm 0.9% off the truth passes as long as the contiguous
    // arm is off the same way. That mattered — during the paged-divergence
    // investigation "the differential passes" was cited as evidence the
    // paged kernel was accurate, which this comparison alone never showed.
    //
    // Both arms store fp16 and fp16 -> fp32 is lossless, so an fp32
    // recomputation over the same tensors is not a third approximation: it
    // is the EXACT answer both arms approximate. Scoring each arm against it
    // makes the differential a three-point comparison that can say WHICH arm
    // is wrong and by how much.

    /// fp32 recomputation of one row's decode attention over the values BOTH
    /// arms were handed.
    ///
    /// Seeded from the same `makeHistories` chunks the two `Arm`s stored and
    /// advanced with each step's new k/v, so it tracks the row without asking
    /// either storage engine anything.
    private struct Fp32Oracle {
        let kind: CBv2LayerKind

        /// `[kvHeads, T, headDim]` fp32.
        private var keys: MLXArray
        private var values: MLXArray

        init(kind: CBv2LayerKind, history: [MLXArray]) {
            self.kind = kind
            var k = MLXArray.zeros([kind.kvHeads, 0, kind.headDim], dtype: .float32)
            var v = k
            for pair in stride(from: 0, to: history.count, by: 2) {
                k = concatenated([k, history[pair].asType(.float32)], axis: 1)
                v = concatenated([v, history[pair + 1].asType(.float32)], axis: 1)
            }
            self.keys = k
            self.values = v
        }

        /// Append one decode step's `[kvHeads, 1, headDim]` k/v.
        mutating func append(keys newKeys: MLXArray, values newValues: MLXArray) {
            keys = concatenated([keys, newKeys.asType(.float32)], axis: 1)
            values = concatenated([values, newValues.asType(.float32)], axis: 1)
        }

        /// The span both arms attend at decode, `[1, kvHeads, T', headDim]`:
        /// the retained ring for a windowed kind (a decode `snapshot()` is
        /// post-eviction, so retained == window), all of history otherwise.
        private var attendable: (keys: MLXArray, values: MLXArray) {
            let t = keys.dim(1)
            var start = 0
            if case .slidingWindow(let window) = kind.attention {
                start = max(0, t - window)
            }
            return (
                keys[0..., start ..< t, 0...].expandedDimensions(axis: 0),
                values[0..., start ..< t, 0...].expandedDimensions(axis: 0))
        }

        /// fp32 attention for one row's `[queryHeads, 1, headDim]` query,
        /// optionally with attendable entry `drop` removed — a planted
        /// storage fault (an entry that never landed, or one a slipped page
        /// bound clobbered).
        func attend(
            queries q: MLXArray, scale: Float, sinks: MLXArray?, dropping drop: Int? = nil
        ) -> MLXArray {
            var (k, v) = attendable
            if let drop {
                k = Self.removingEntry(drop, from: k)
                v = Self.removingEntry(drop, from: v)
            }
            return PagedAttentionReference.composedAttention(
                queries: q.asType(.float32).expandedDimensions(axis: 0),
                keys: k, values: v, scale: scale, sinks: sinks)
        }

        /// Per-query-head softmax weights over the attendable span,
        /// `[queryHeads, T']`. Sinks are deliberately ignored: a sink scales
        /// every column of its head by one denominator factor, so it cannot
        /// change which key is heaviest or reorder the weights.
        private func attentionWeights(queries q: MLXArray, scale: Float) -> MLXArray {
            let (k, _) = attendable
            let gqa = kind.queryHeads / kind.kvHeads
            let broadcastK = gqa > 1 ? repeated(k, count: gqa, axis: 1) : k
            let scores = matmul(
                q.asType(.float32).expandedDimensions(axis: 0) * scale,
                broadcastK.transposed(0, 1, 3, 2))
            return softmax(scores, axis: -1, precise: true)
                .reshaped([kind.queryHeads, broadcastK.dim(2)])
        }

        /// Index of the key carrying the most softmax mass across query
        /// heads — the live entry a storage fault would hurt most.
        func heaviestKeyIndex(queries q: MLXArray, scale: Float) -> Int {
            Int(
                argMax(attentionWeights(queries: q, scale: scale).sum(axis: 0), axis: -1)
                    .item(Int32.self))
        }

        /// Largest weight ANY query head puts on the OLDEST attendable key.
        func oldestKeyWeight(queries q: MLXArray, scale: Float) -> Float {
            attentionWeights(queries: q, scale: scale)[0..., 0].max().item(Float.self)
        }

        private static func removingEntry(_ index: Int, from a: MLXArray) -> MLXArray {
            let t = a.dim(2)
            precondition(index >= 0 && index < t, "planted fault index \(index) outside [0, \(t))")
            if index == 0 { return a[0..., 0..., 1 ..< t, 0...] }
            if index == t - 1 { return a[0..., 0..., 0 ..< (t - 1), 0...] }
            return concatenated(
                [a[0..., 0..., 0 ..< index, 0...], a[0..., 0..., (index + 1) ..< t, 0...]],
                axis: 2)
        }
    }

    /// `‖got − reference‖₂ / ‖reference‖₂` over a whole row output.
    ///
    /// Scale- and shape-free, so one number is comparable across the d64 and
    /// d512 shapes and answers "how far from the truth is this arm" — which
    /// `allClose`'s pass/fail cannot.
    private static func relativeError(_ got: MLXArray, vs reference: MLXArray) -> Float {
        let g = got.asType(.float32)
        let r = reference.asType(.float32)
        let residual = (g - r).square().sum().item(Float.self)
        let magnitude = r.square().sum().item(Float.self)
        return (magnitude > 0 ? residual / magnitude : residual).squareRoot()
    }

    /// Absolute ceiling on the CONTIGUOUS arm's own distance from the fp32
    /// truth.
    ///
    /// `oracleBar` is expressed relative to this arm, so if it drifts the bar
    /// drifts with it and a systematic error BOTH arms share rides straight
    /// through — the exact blindness that made the old backends-to-each-other
    /// comparison unable to say anything about accuracy. This is the
    /// precondition that keeps the relative bar meaningful, and it caps the
    /// shared-error hole at 1%. Measured 2e-4 to 2.8e-3 across the shapes in
    /// `differentialOracleRejectsPlantedFaults`.
    private static let referenceArmCeiling: Float = 1e-2

    /// The bar on the paged arm: up to 3x the contiguous arm's own distance
    /// from the fp32 truth, with a 1e-2 floor.
    ///
    /// Both halves earn their place. The 3x rides on the contiguous arm so
    /// the bar tracks each row's actual conditioning rather than a constant
    /// chosen for the easiest shape; it is the clause that binds on a
    /// badly-conditioned shape where fp16 attention is simply hard. The floor
    /// stops a row where the contiguous arm happens to land near-exact from
    /// turning into a bitwise demand on a differently-implemented kernel.
    ///
    /// What this admits, stated plainly: up to 1% absolute paged error. That
    /// is a bounded claim about ACCURACY, where the old two-point bar made no
    /// accuracy claim at all — two arms wrong by the same 50% agreed with
    /// each other perfectly. Measured paged error today is ~2e-4, i.e. ~50x
    /// inside the floor, and the floor is the clause that binds at every
    /// shape in `differentialOracleRejectsPlantedFaults`.
    private static func oracleBar(contiguous: Float) -> Float {
        max(3 * contiguous, 1e-2)
    }

    /// The full oracle verdict for one row, as a predicate — the same two
    /// conditions `expectPagedWithinOracleBar` asserts, so the calibration
    /// test can show a fault the verdict REJECTS that `assertClose` accepts.
    private static func oracleAccepts(
        paged: MLXArray, contiguous: MLXArray, reference: MLXArray
    ) -> Bool {
        let ec = relativeError(contiguous, vs: reference)
        return ec <= referenceArmCeiling
            && relativeError(paged, vs: reference) <= oracleBar(contiguous: ec)
    }

    /// Score both arms against the fp32 reference and hold each to its bar.
    @discardableResult
    private func expectPagedWithinOracleBar(
        paged: MLXArray, contiguous: MLXArray, reference: MLXArray, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) -> (paged: Float, contiguous: Float, bar: Float) {
        let ep = Self.relativeError(paged, vs: reference)
        let ec = Self.relativeError(contiguous, vs: reference)
        let bar = Self.oracleBar(contiguous: ec)
        #expect(
            ec <= Self.referenceArmCeiling,
            "\(what): the CONTIGUOUS arm is \(ec) from the fp32 reference, past \(Self.referenceArmCeiling) — the paged bar rides on this arm, so it means nothing until this holds",
            sourceLocation: sourceLocation)
        #expect(
            ep <= bar,
            "\(what): paged relErr \(ep) vs the fp32 reference exceeds \(bar) (contiguous arm \(ec))",
            sourceLocation: sourceLocation)
        return (ep, ec, bar)
    }

    /// Decode parity between the two STORAGE ENGINES across the shape matrix
    /// `decodeKernelParity` covers with a mirror, plus both gemma-4-26b layer
    /// shapes (sliding: window 1024 / d256 / 8 kv heads; full: d512 / 2 kv
    /// heads).
    ///
    /// Two independent bars per row, because they catch different things:
    ///
    /// 1. STORAGE — the two arms against each other at `1e-2 / 2e-3`
    ///    (`assertClose`). Agreement here requires both engines to have
    ///    stored, evicted and re-read the same values; a mis-sized ring or a
    ///    slipped page bound breaks it whatever the numerics say.
    /// 2. ACCURACY — each arm against `Fp32Oracle`, the exact answer over the
    ///    same stored values: the contiguous arm to `referenceArmCeiling`,
    ///    the paged arm to `oracleBar` on top of it. Bar 1 is blind to any
    ///    error the two arms SHARE, at any magnitude; this one is not.
    ///
    /// Neither subsumes the other, so both run.
    /// `differentialOracleRejectsPlantedFaults` calibrates bar 2 and exhibits
    /// a fault bar 1 accepts and bar 2 rejects.
    @Test(arguments: [
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: true),
        (headDim: 64, kvHeads: 2, queryHeads: 16, window: Int?(20), sinks: true),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?(33), sinks: false),
        (headDim: 256, kvHeads: 8, queryHeads: 16, window: Int?(1024), sinks: false),
        (headDim: 512, kvHeads: 2, queryHeads: 16, window: Int?.none, sinks: false),
    ])
    func decodeMatchesContiguousBackend(
        _ shape: (headDim: Int, kvHeads: Int, queryHeads: Int, window: Int?, sinks: Bool)
    ) throws {
        var rng = KeyedRandom(seed: 0x5EED_0A11)
        let attention: CBv2LayerKind.Attention =
            shape.window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, hasSinks: shape.sinks, headDim: shape.headDim,
            kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)

        // Row lengths straddle a page boundary (16), a whole page, and — for
        // the window-20/33 kinds — the window itself, so the ring has already
        // wrapped on at least one row before the first differential step.
        let lengths = [4, 16, 33, 100]
        let histories = makeHistories(kind: kind, lengths: lengths, rng: &rng)
        let b = lengths.count
        let scale = Float(1.0 / Double(shape.headDim).squareRoot())
        // Sinks in the QUERY dtype: both arms must hold bit-identical inputs
        // for a storage differential to mean anything, and the paged decode
        // kernel re-widens to fp32 in `preparedSinks` regardless.
        //
        // This line used to be load-bearing for a different reason, worth one
        // note so nobody "simplifies" it back. On its first run this test
        // aborted the whole bundle: `PagedLayerCache` coerced sinks to the
        // query dtype, `CBv2AttentionV1` did not, and fp16 queries with fp32
        // sinks reached MLXFast and raised "[scaled_dot_product_attention]
        // Type of sinks must promote to output type float16" — a `fatalError`,
        // i.e. a daemon abort, on the CONTIGUOUS backend only. Latent in
        // production only because gpt-oss-20b's `sinks` parameter happens to
        // load in the activation dtype. Fixed 2026-07-25 (A2): AttentionV1 now
        // coerces at both MLXFast sites, and cross-backend sink-dtype
        // acceptance has its own cover in `CBv2AttentionSinkDtypeTests`.
        // Storage parity — this test — is not the place to re-assert it.
        let sinks: MLXArray? = shape.sinks ? rng.normal([shape.queryHeads]) : nil

        let paged = try Arm(.paged, kind: kind, histories: histories, maxLength: 2048)
        let contiguous = try Arm(
            .contiguous, kind: kind, histories: histories, maxLength: 2048)
        var oracles = histories.map { Fp32Oracle(kind: kind, history: $0) }

        for step in 0 ..< 4 {
            let q = rng.normal([b, shape.queryHeads, 1, shape.headDim])
            let k = rng.normal([b, shape.kvHeads, 1, shape.headDim])
            let v = rng.normal([b, shape.kvHeads, 1, shape.headDim])
            let outPaged = paged.step(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            let outContiguous = contiguous.step(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            #expect(outPaged.shape == [b, shape.queryHeads, 1, shape.headDim])
            for row in 0 ..< b {
                let got = outPaged[row].expandedDimensions(axis: 0)
                let want = outContiguous[row].expandedDimensions(axis: 0)
                assertClose(got, want)
                oracles[row].append(keys: k[row], values: v[row])
                let reference = oracles[row].attend(
                    queries: q[row], scale: scale, sinks: sinks)
                expectPagedWithinOracleBar(
                    paged: got, contiguous: want, reference: reference,
                    "d\(shape.headDim) kv\(shape.kvHeads) q\(shape.queryHeads) "
                        + "window \(shape.window.map(String.init) ?? "none") "
                        + "row \(row) step \(step)")
            }
        }
    }

    /// One planted-fault calibration case.
    struct OracleProbe: Sendable, CustomStringConvertible {
        let label: String
        let headDim: Int
        let kvHeads: Int
        let queryHeads: Int
        let window: Int?
        let scale: Float
        let history: Int
        /// Whether dropping the OLDEST key is a discriminating probe at this
        /// shape — see the blind spot documented on the test below.
        let oldestKeyIsSensitive: Bool

        var description: String { "\(label) history \(history)" }
    }

    /// Calibration for `oracleBar`: show, in the test itself, that the band
    /// actually separates honest paged output from a broken store.
    ///
    /// A tolerance nobody has probed is a wish. This runs the SAME instrument
    /// `decodeMatchesContiguousBackend` uses, over the shapes that motivated
    /// it, on honest output AND on deliberately corrupted output.
    ///
    /// Measured 2026-07-25 (M4 Max, seed below), relErr vs the fp32
    /// reference — reproduce it, do not extend trust to the table. `@prod`
    /// rows are the same shape at the production query scale
    /// `1/sqrt(headDim)`; the others use the scale 1.0 the divergence
    /// investigation ran at, which for d512 is 23x production.
    ///
    ///     probe                      paged    contig   paged/contig  agree
    ///     gemma4-sliding      h33    1.8e-4   1.8e-4   1.00          yes
    ///     gemma4-sliding      h200   2.0e-4   2.0e-4   1.00          yes
    ///     gemma4-full         h33    1.6e-4   1.1e-3   0.14          NO
    ///     gemma4-full         h200   1.7e-4   2.8e-3   0.06          NO
    ///     gemma4-sliding@prod h33    2.0e-4   2.0e-4   1.00          yes
    ///     gemma4-sliding@prod h200   2.1e-4   2.1e-4   1.00          yes
    ///     gemma4-full@prod    h33    2.1e-4   4.7e-4   0.45          yes
    ///     gemma4-full@prod    h200   2.1e-4   5.4e-4   0.39          yes
    ///     gptoss              h33    2.1e-4   2.1e-4   1.00          yes
    ///     gptoss              h200   2.0e-4   2.0e-4   1.00          yes
    ///
    ///     probe                      dropOldest   dropHeaviest   oldest wt
    ///     gemma4-sliding      h33    4.6e-2       4.5e-1         1.2e-1
    ///     gemma4-sliding      h200   3.9e-1       4.7e-1         9.8e-1
    ///     gemma4-full         h33    4.4e-8  (!)  4.9e-1         1.2e-8
    ///     gemma4-full         h200   4.1e-8  (!)  3.7e-1         3.8e-11
    ///     gemma4-sliding@prod h33    1.6e-1       2.4e-1         1.1e-1
    ///     gemma4-sliding@prod h200   1.5e-1       2.2e-1         5.6e-2
    ///     gemma4-full@prod    h33    1.1e-1       2.4e-1         6.4e-2
    ///     gemma4-full@prod    h200   5.7e-2       1.6e-1         1.7e-2
    ///     gptoss              h33    1.4e-1       3.6e-1         6.6e-2
    ///     gptoss              h200   5.2e-2       2.6e-1         1.5e-2
    ///
    /// That is the separation the bar rides on: honest output sits at ~2e-4,
    /// the bar at 1e-2, a planted fault at 4.6e-2 to 4.9e-1 — 220x to 3000x
    /// above honest, 4.6x to 49x above the bar. The paged arm is never worse
    /// than the contiguous one anywhere in the matrix, and on gemma-4 full it
    /// is 2.2x more accurate at the production scale and 7-17x more accurate
    /// at scale 1.0. On that shape the CONTIGUOUS arm is the loose one.
    ///
    /// KNOWN BLIND SPOT, encoded rather than papered over — the `(!)` rows.
    /// Dropping the OLDEST key is not a discriminating probe on gemma-4 FULL
    /// AT SCALE 1.0: the softmax is peaked hard enough that the oldest key
    /// carries ~1e-8 of the mass, so removing it does not move the output at
    /// all. Those two cases carry `oldestKeyIsSensitive: false` and the test
    /// pins the CAUSE — the oldest key's weight — rather than asserting a
    /// fault it demonstrably cannot see. Coverage there comes from the second
    /// fault, dropping the HEAVIEST key, which discriminates everywhere by
    /// construction.
    ///
    /// The `@prod` rows are what bound that gap, and they are the reason both
    /// scales are in the matrix. At `1/sqrt(512)` the same shape's oldest key
    /// carries 6.4e-2 / 1.7e-2 of the mass and the same fault scores 1.1e-1 /
    /// 5.7e-2 — comfortably discriminating. So the blind spot is a property
    /// of an extreme fixture, not of the shape the model actually runs, and
    /// it does not exist at production scale. Do not delete the scale-1.0
    /// rows to make the carve-out go away: they are the stress case, they
    /// reproduce the divergence investigation's own measurement, and they are
    /// where the paged kernel's accuracy advantage is largest.
    ///
    /// All three planted faults are real failure modes: an entry the ring
    /// evicted one step early, a live entry a slipped page bound clobbered,
    /// and — the one this whole rework exists for — a systematic error BOTH
    /// arms share, which `assertClose(paged, contiguous)` accepts at any
    /// magnitude and the fp32 reference rejects past 1%.
    ///
    /// One finding worth carrying forward, because it is the wave's thesis in
    /// miniature: the `agree` column. On the two scale-1.0 gemma-4-full
    /// probes the arms breach the two-point bar against EACH OTHER with
    /// nothing planted — worst elementwise `|a-b| / (atol + rtol*|b|)` is 3.4
    /// at history 33 and 7.1 at history 200, against 0.0-0.05 everywhere
    /// else. The fp32 reference is what makes that legible: it attributes the
    /// gap to the CONTIGUOUS arm, and the test asserts that attribution.
    /// Without the third point the same observation is just "the backends
    /// diverge", which is exactly how this got misread as a paged defect.
    ///
    /// `decodeMatchesContiguousBackend` is unaffected: it drives d512 at
    /// `1/sqrt(512)`, where the arms do agree, and its `assertClose` passes
    /// on every shape it covers.
    @Test(arguments: [
        // gemma-4-26b sliding layer: d256, GQA 2, window 1024.
        OracleProbe(
            label: "gemma4-sliding", headDim: 256, kvHeads: 8, queryHeads: 16,
            window: 1024, scale: 1.0, history: 33, oldestKeyIsSensitive: true),
        OracleProbe(
            label: "gemma4-sliding", headDim: 256, kvHeads: 8, queryHeads: 16,
            window: 1024, scale: 1.0, history: 200, oldestKeyIsSensitive: true),
        // gemma-4-26b full layer: d512, GQA 8.
        OracleProbe(
            label: "gemma4-full", headDim: 512, kvHeads: 2, queryHeads: 16,
            window: nil, scale: 1.0, history: 33, oldestKeyIsSensitive: false),
        OracleProbe(
            label: "gemma4-full", headDim: 512, kvHeads: 2, queryHeads: 16,
            window: nil, scale: 1.0, history: 200, oldestKeyIsSensitive: false),
        // The SAME two gemma-4 shapes at the production query scale,
        // `1/sqrt(headDim)`. Scale 1.0 at d512 is 23x that and peaks the
        // softmax far harder than the model ever does, which is what creates
        // the oldest-key blind spot above. Running both scales says whether
        // the gap is a property of the oracle or of an unrealistic fixture —
        // the answer is in the second table on this test.
        OracleProbe(
            label: "gemma4-sliding@prod", headDim: 256, kvHeads: 8, queryHeads: 16,
            window: 1024, scale: Float(1.0 / Double(256).squareRoot()), history: 33,
            oldestKeyIsSensitive: true),
        OracleProbe(
            label: "gemma4-sliding@prod", headDim: 256, kvHeads: 8, queryHeads: 16,
            window: 1024, scale: Float(1.0 / Double(256).squareRoot()), history: 200,
            oldestKeyIsSensitive: true),
        OracleProbe(
            label: "gemma4-full@prod", headDim: 512, kvHeads: 2, queryHeads: 16,
            window: nil, scale: Float(1.0 / Double(512).squareRoot()), history: 33,
            oldestKeyIsSensitive: true),
        OracleProbe(
            label: "gemma4-full@prod", headDim: 512, kvHeads: 2, queryHeads: 16,
            window: nil, scale: Float(1.0 / Double(512).squareRoot()), history: 200,
            oldestKeyIsSensitive: true),
        // gpt-oss decode shape: d64, scale 0.125.
        OracleProbe(
            label: "gptoss", headDim: 64, kvHeads: 2, queryHeads: 8,
            window: nil, scale: 0.125, history: 33, oldestKeyIsSensitive: true),
        OracleProbe(
            label: "gptoss", headDim: 64, kvHeads: 2, queryHeads: 8,
            window: nil, scale: 0.125, history: 200, oldestKeyIsSensitive: true),
    ])
    func differentialOracleRejectsPlantedFaults(_ probe: OracleProbe) throws {
        var rng = KeyedRandom(seed: 0x5EED_0A14)
        let attention: CBv2LayerKind.Attention =
            probe.window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, headDim: probe.headDim,
            kvHeads: probe.kvHeads, queryHeads: probe.queryHeads)
        let histories = makeHistories(kind: kind, lengths: [probe.history], rng: &rng)
        let paged = try Arm(.paged, kind: kind, histories: histories, maxLength: 2048)
        let contiguous = try Arm(
            .contiguous, kind: kind, histories: histories, maxLength: 2048)
        var oracle = Fp32Oracle(kind: kind, history: histories[0])

        let q = rng.normal([1, probe.queryHeads, 1, probe.headDim])
        let k = rng.normal([1, probe.kvHeads, 1, probe.headDim])
        let v = rng.normal([1, probe.kvHeads, 1, probe.headDim])
        let outPaged = paged.step(
            queries: q, keys: k, values: v, scale: probe.scale, sinks: nil)
        let outContiguous = contiguous.step(
            queries: q, keys: k, values: v, scale: probe.scale, sinks: nil)
        oracle.append(keys: k[0], values: v[0])
        let reference = oracle.attend(queries: q[0], scale: probe.scale, sinks: nil)

        // 1. Honest output clears the bar.
        let honest = expectPagedWithinOracleBar(
            paged: outPaged, contiguous: outContiguous, reference: reference,
            "\(probe) honest")

        // 2. Planted storage faults, scored with the SAME instrument.
        let heaviest = oracle.heaviestKeyIndex(queries: q[0], scale: probe.scale)
        let oldestErr = Self.relativeError(
            oracle.attend(queries: q[0], scale: probe.scale, sinks: nil, dropping: 0),
            vs: reference)
        let heaviestErr = Self.relativeError(
            oracle.attend(queries: q[0], scale: probe.scale, sinks: nil, dropping: heaviest),
            vs: reference)

        expectPlantedFaultRejected(
            "\(probe): oldest key dropped", error: oldestErr, honest: honest,
            enforced: probe.oldestKeyIsSensitive)
        expectPlantedFaultRejected(
            "\(probe): heaviest key (index \(heaviest)) dropped", error: heaviestErr,
            honest: honest, enforced: true)
        if !probe.oldestKeyIsSensitive {
            // Pin the CAUSE of the blind spot, not the symptom. If the oldest
            // key ever starts carrying real mass at this shape the probe
            // becomes usable — flip `oldestKeyIsSensitive`, do not widen
            // anything.
            let oldestWeight = oracle.oldestKeyWeight(queries: q[0], scale: probe.scale)
            #expect(
                oldestWeight < 1e-3,
                "\(probe): oldest key carries weight \(oldestWeight) — no longer negligible, so `oldestKeyIsSensitive` should be true here"
            )
        }

        // 3. The error class the backends-to-each-other bar cannot see AT
        //    ALL: one both arms SHARE.
        //
        //    Modelled as its limiting case — both arms return the SAME answer
        //    and it is 5% off the truth. Bar 1's verdict on that pair is
        //    `arraysAgree(x, x)`: true with residual exactly zero, on every
        //    shape, at ANY drift magnitude, because bar 1 only ever asks
        //    whether the arms match EACH OTHER. That half needs no assertion
        //    — it is an identity, and an identity is a stronger statement
        //    than a passing sample. Bar 2 scores each arm against ground
        //    truth, so it rejects the same pair outright. The accept -> reject
        //    flip against the honest pair just above is the discrimination
        //    the two-point bar never had.
        //
        //    Do NOT "improve" this into scaling the two DISTINCT arm outputs
        //    and checking bar 1 still passes. `allClose` is not
        //    scale-invariant: `|a-b| <= atol + rtol*|b|` scales fully on the
        //    left and only partly on the right, so a common scaling makes the
        //    comparison strictly HARDER. Worse, on the two scale-1.0
        //    gemma-4-full probes the arms do not clear that bar even
        //    UNDRIFTED — worst elementwise `|a-b| / (atol + rtol*|b|)` is 3.4
        //    at h33 and 7.1 at h200, against 0.0 to 0.05 on the other eight.
        //    Both halves of such a test are then false, and phrasing it as
        //    `verdictBefore == verdictAfter` makes it pass vacuously on
        //    exactly the shapes that matter most.
        let sharedlyWrong = outContiguous * MLXArray(Float(1.05))
        #expect(
            Self.oracleAccepts(
                paged: outPaged, contiguous: outContiguous, reference: reference),
            "\(probe): the fp32 oracle rejected HONEST output — the flip below would prove nothing"
        )
        #expect(
            !Self.oracleAccepts(
                paged: sharedlyWrong, contiguous: sharedlyWrong, reference: reference),
            "\(probe): the fp32 oracle accepted an answer both arms agreed on and both got 5% wrong — it has lost the one thing the two-point bar never had"
        )

        // 4. Where the two arms genuinely disagree ELEMENTWISE, the third
        //    point says which of them is wrong — the wave's root-cause claim
        //    reduced to an assertion. On gemma-4-full at scale 1.0 they
        //    breach `1e-2 / 2e-3` against each other, and it is the CONTIGUOUS
        //    arm that is off: fp16 SDPA over a 200-key history at d512 with a
        //    peaked softmax, against the paged kernel's fp32 accumulation.
        //    Reading that disagreement as a paged defect is precisely the
        //    mistake this whole rework exists to prevent, so pin the
        //    attribution rather than the agreement.
        if !Self.arraysAgree(outPaged, outContiguous) {
            #expect(
                honest.contiguous > honest.paged,
                "\(probe): the arms disagree past the two-point bar and the fp32 reference blames the PAGED arm (paged \(honest.paged), contiguous \(honest.contiguous)) — that is a real paged regression, not the known contiguous looseness"
            )
        }
    }

    /// A planted fault must land outside the bar AND nowhere near the honest
    /// paged arm — two orders of magnitude of daylight, not a hair.
    ///
    /// `enforced: false` skips both assertions. It is used ONLY for the
    /// documented gemma-4-full blind spot, where this probe is measurably
    /// unable to bite; the caller asserts the reason (the oldest key's
    /// weight) in its place, so the gap is stated, never silent.
    private func expectPlantedFaultRejected(
        _ name: String, error: Float,
        honest: (paged: Float, contiguous: Float, bar: Float), enforced: Bool,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        guard enforced else { return }
        #expect(
            error > honest.bar,
            "\(name): scores \(error), inside the bar \(honest.bar) — the oracle cannot see this fault",
            sourceLocation: sourceLocation)
        #expect(
            error > 100 * honest.paged,
            "\(name): scores \(error), within 100x of honest paged output \(honest.paged) — no separation",
            sourceLocation: sourceLocation)
    }

    /// The chunked-prefill fallback compared against the contiguous engine
    /// rather than a mirror. Prefill is per-request `[1, chunk]` on BOTH
    /// backends, so one call sequence drives both.
    @Test(arguments: [Int?.none, Int?(24)])
    func prefillMatchesContiguousBackend(window: Int?) throws {
        MLXRandom.seed(0x5EED_0A12)
        let attention: CBv2LayerKind.Attention =
            window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, headDim: 64, kvHeads: 2, queryHeads: 4)
        let scale: Float = 0.125

        let paged = try Arm(.paged, kind: kind, histories: [[]], maxLength: 512)
        let contiguous = try Arm(.contiguous, kind: kind, histories: [[]], maxLength: 512)

        // Chunks that cross the window (24) and page boundaries (16), then a
        // decode step over the same ring — the prefill -> decode transition is
        // where a mis-sized ring first shows up.
        for chunk in [9, 32, 16, 1] {
            let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
            let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            assertClose(
                paged.step(queries: q, keys: k, values: v, scale: scale, sinks: nil),
                contiguous.step(queries: q, keys: k, values: v, scale: scale, sinks: nil))
        }
    }

    /// The 200-step greedy trajectory with the reference arm backed by a real
    /// contiguous store instead of an in-file recomputation. Divergence
    /// compounds: one wrong key at step 3 changes every later token, so a
    /// token-match bar over 200 steps is a far sharper storage oracle than any
    /// single-step comparison. Windowed(16): the ring wraps ~12 times, so
    /// eviction arithmetic is under test, not just append arithmetic.
    @Test func greedyDecodeMatchesContiguousBackend200Steps() throws {
        MLXRandom.seed(0x5EED_0A13)
        let vocab = 97
        let headDim = 64
        let kvHeads = 2
        let queryHeads = 4

        let embedQ = MLXRandom.normal([vocab, queryHeads * headDim], dtype: .float16)
        let embedK = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let embedV = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let unembed = MLXRandom.normal([queryHeads * headDim, vocab], dtype: .float16)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        let kind = CBv2LayerKind(
            attention: .slidingWindow(16), headDim: headDim,
            kvHeads: kvHeads, queryHeads: queryHeads)
        let paged = try Arm(.paged, kind: kind, histories: [[]], maxLength: 512)
        let contiguous = try Arm(.contiguous, kind: kind, histories: [[]], maxLength: 512)

        func project(_ token: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
            (
                embedQ[token].reshaped([1, queryHeads, 1, headDim]),
                embedK[token].reshaped([1, kvHeads, 1, headDim]),
                embedV[token].reshaped([1, kvHeads, 1, headDim])
            )
        }
        func logits(_ attnOut: MLXArray) -> MLXArray {
            matmul(
                attnOut.reshaped([1, queryHeads * headDim]).asType(.float32),
                unembed.asType(.float32))
        }

        var pagedToken = 1
        var contiguousToken = 1
        var matches = 0
        let steps = 200
        for _ in 0 ..< steps {
            let p = project(pagedToken)
            let nextPaged = argMax(
                logits(
                    paged.step(queries: p.q, keys: p.k, values: p.v, scale: scale, sinks: nil)),
                axis: -1
            ).item(Int32.self)

            let c = project(contiguousToken)
            let nextContiguous = argMax(
                logits(
                    contiguous.step(
                        queries: c.q, keys: c.k, values: c.v, scale: scale, sinks: nil)),
                axis: -1
            ).item(Int32.self)

            if nextPaged == nextContiguous { matches += 1 }
            pagedToken = Int(nextPaged)
            contiguousToken = Int(nextContiguous)
        }
        #expect(
            Double(matches) >= 0.995 * Double(steps),
            "paged vs contiguous greedy token match \(matches)/\(steps) below 99.5%")
    }
}
