// CBv2AttentionSinkDtypeTests.swift
//
// Sink-dtype acceptance for the CONTIGUOUS attention path.
//
// MLX's fused SDPA requires the sink dtype to PROMOTE to the output dtype:
// fp16 queries with fp32 sinks raise
//
//     [scaled_dot_product_attention] Type of sinks must promote to output
//     type float16
//
// as a `fatalError` inside MLX — a process abort, not a throw, so a daemon
// hosting several models loses every in-flight request and emits nothing.
//
// `PagedLayerCache` has always coerced (`sinks?.asType(queries.dtype)` on the
// SDPA prefill, `.asType(.float32)` in `preparedSinks` for the decode kernel),
// so the paged backend accepts fp16 OR fp32 sinks. `CBv2AttentionV1` did not,
// which made the two backends disagree about their input domain with the
// contiguous one — the backend shipping today — aborting the process.
//
// These tests pin the contiguous path's acceptance of a WIDER sink than the
// activations, on every route that reaches MLXFast SDPA:
//
//   `attend`          — decode, unblocked prefill, query-blocked prefill,
//                       serialized (MTP) prefill;
//   `attendSpanChunk` — vision span-bearing prefill chunks.
//
// The softcap sibling of each site goes to `PagedAttentionReference
// .composedAttention`, which runs in fp32 throughout and casts sinks itself,
// so it was never exposed and is covered here only as a reference oracle.
//
// Method: sinks are generated in fp16 and widened with `.asType(.float32)`,
// so the fp16 and fp32 arms carry BIT-IDENTICAL VALUES. Any divergence is
// therefore a property of the dtype path, not of rounding, which lets the
// same-backend comparisons assert exact equality.
//
// No in-bundle NEGATIVE test. MLX throws `std::invalid_argument` in C++,
// mlx-c routes it to the installed error handler, and mlx-swift's handler
// calls `fatalError` (`MLX/ErrorHandler.swift`) — SIGTRAP, uncatchable, so
// an uncoerced terminal takes the whole bundle down instead of recording an
// issue. Observing the pre-fix trap therefore needs a deliberately-aborting
// child process, i.e. an extra executable target in the shared
// Package.swift. The positive tests below DO still fail without the fix,
// just as a bundle crash rather than a recorded failure.
// `sinkCoercionIsSharedContract` additionally pins the coercion helper
// itself, which fails CLEANLY and is the assertion that survives to say
// what went wrong.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

/// Live query-block width. Read once, because `CBv2AttentionV1
/// .queryBlockSize` is a lazily-initialized `static let` fixed for the
/// process — every chunk length below is derived from it rather than from
/// the 128 default, so this suite keeps testing the routes it names under
/// `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK` (including the `0` kill switch).
private let liveBlock = CBv2AttentionV1.queryBlockSize

@Suite("CBv2AttentionSinkDtype", .serialized)
struct CBv2AttentionSinkDtypeTests {

    // MARK: - Fixtures

    private enum StorageArm {
        case paged
        case contiguous
    }

    /// One layer of one storage engine behind `CBv2AttendingLayerCache`, fed
    /// history in identical chunks so both arms observe identical
    /// `lastUpdateTokens` trajectories. Mirrors the `Arm` in
    /// `CBv2PagedKernelTests` (private there, so not reusable).
    private final class Arm {
        let cache: CBv2AttendingLayerCache
        let rows: [CBv2SequenceKV]
        private let release: () -> Void

        init(
            _ arm: StorageArm, kind: CBv2LayerKind, histories: [[MLXArray]],
            maxLength: Int, maxPrefillChunk: Int = 64
        ) throws {
            let kinds = [kind]
            var rows: [CBv2SequenceKV] = []
            switch arm {
            case .paged:
                let backend = try PagedKVBackend(
                    layerKinds: kinds,
                    config: PagedKVPoolConfig(
                        capacityBytes: 128 << 20,
                        maxPrefillChunk: maxPrefillChunk,
                        nominalMaxSequenceLength: 4096))
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
            self.rows = rows
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

    /// `[kvHeads, n, headDim]` history chunk pairs per row, generated ONCE so
    /// every arm stores byte-identical bytes.
    private func makeHistories(
        kind: CBv2LayerKind, lengths: [Int], chunk: Int = 64
    ) -> [[MLXArray]] {
        lengths.map { tokens in
            var chunks: [MLXArray] = []
            var remaining = tokens
            while remaining > 0 {
                let n = min(remaining, chunk)
                chunks.append(MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16))
                chunks.append(MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16))
                remaining -= n
            }
            return chunks
        }
    }

    /// fp16 sinks and their exactly-equal fp32 widening — the pair every test
    /// compares. `[queryHeads]`, the shape a real model's sink parameter has.
    private func sinkPair(queryHeads: Int) -> (narrow: MLXArray, wide: MLXArray) {
        let narrow = MLXRandom.normal([queryHeads], dtype: .float16)
        eval(narrow)
        return (narrow, narrow.asType(.float32))
    }

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
            Issue.record(
                "arrays differ, max abs err \(diff)", sourceLocation: sourceLocation)
        }
    }

    private func assertIdentical(
        _ got: MLXArray, _ want: MLXArray,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        #expect(
            (got .== want).all().item(Bool.self),
            "widening the sink changed the result", sourceLocation: sourceLocation)
    }

    // MARK: - Decode: the shipping hot path

    /// fp32 sinks against fp16 queries through the CONTIGUOUS decode path
    /// must (a) not abort, and (b) agree with the paged backend, which
    /// accepts the same input natively.
    ///
    /// Both windowed and full layers, and both a GQA-1 and a GQA-4 shape, so
    /// the cast is exercised on every head layout the sinks model uses.
    @Test(arguments: [
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none),
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?(20)),
        (headDim: 64, kvHeads: 4, queryHeads: 4, window: Int?.none),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?(33)),
    ])
    func decodeAcceptsWiderSinksAndMatchesPaged(
        _ shape: (headDim: Int, kvHeads: Int, queryHeads: Int, window: Int?)
    ) throws {
        MLXRandom.seed(0x51_4B_D7_01)
        let attention: CBv2LayerKind.Attention =
            shape.window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, hasSinks: true, headDim: shape.headDim,
            kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)

        // Straddle the page boundary (16), a whole page, and the window.
        let lengths = [4, 16, 33, 100]
        let histories = makeHistories(kind: kind, lengths: lengths)
        let b = lengths.count
        let scale = Float(1.0 / Double(shape.headDim).squareRoot())
        let (narrowSinks, wideSinks) = sinkPair(queryHeads: shape.queryHeads)

        let paged = try Arm(.paged, kind: kind, histories: histories, maxLength: 2048)
        let contiguous = try Arm(
            .contiguous, kind: kind, histories: histories, maxLength: 2048)
        // Third arm, fp16 sinks, driven with the SAME inputs: isolates the
        // cast's numeric effect on the contiguous path from any cross-backend
        // difference.
        let contiguousNarrow = try Arm(
            .contiguous, kind: kind, histories: histories, maxLength: 2048)

        for _ in 0 ..< 4 {
            let q = MLXRandom.normal([b, shape.queryHeads, 1, shape.headDim], dtype: .float16)
            let k = MLXRandom.normal([b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let v = MLXRandom.normal([b, shape.kvHeads, 1, shape.headDim], dtype: .float16)

            let outPaged = paged.step(
                queries: q, keys: k, values: v, scale: scale, sinks: wideSinks)
            let outContiguous = contiguous.step(
                queries: q, keys: k, values: v, scale: scale, sinks: wideSinks)
            let outNarrow = contiguousNarrow.step(
                queries: q, keys: k, values: v, scale: scale, sinks: narrowSinks)

            #expect(outContiguous.shape == [b, shape.queryHeads, 1, shape.headDim])
            #expect(outContiguous.dtype == .float16)
            assertIdentical(outContiguous, outNarrow)
            for row in 0 ..< b {
                assertClose(
                    outContiguous[row].expandedDimensions(axis: 0),
                    outPaged[row].expandedDimensions(axis: 0))
            }
        }
    }

    // MARK: - Prefill: chunked, unblocked

    /// `[1, chunk]` prefill on both backends with fp32 sinks. Chunks cross the
    /// window (24) and the page boundary (16), then a decode step over the
    /// same ring.
    @Test(arguments: [Int?.none, Int?(24)])
    func prefillAcceptsWiderSinksAndMatchesPaged(window: Int?) throws {
        MLXRandom.seed(0x51_4B_D7_02)
        let attention: CBv2LayerKind.Attention =
            window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, hasSinks: true, headDim: 64, kvHeads: 2, queryHeads: 8)
        let scale: Float = 0.125
        let (narrowSinks, wideSinks) = sinkPair(queryHeads: 8)

        let paged = try Arm(.paged, kind: kind, histories: [[]], maxLength: 512)
        let contiguous = try Arm(.contiguous, kind: kind, histories: [[]], maxLength: 512)
        let contiguousNarrow = try Arm(
            .contiguous, kind: kind, histories: [[]], maxLength: 512)

        for chunk in [9, 32, 16, 1] {
            let q = MLXRandom.normal([1, 8, chunk, 64], dtype: .float16)
            let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)

            let outPaged = paged.step(
                queries: q, keys: k, values: v, scale: scale, sinks: wideSinks)
            let outContiguous = contiguous.step(
                queries: q, keys: k, values: v, scale: scale, sinks: wideSinks)
            let outNarrow = contiguousNarrow.step(
                queries: q, keys: k, values: v, scale: scale, sinks: narrowSinks)

            #expect(outContiguous.shape == [1, 8, chunk, 64])
            assertIdentical(outContiguous, outNarrow)
            assertClose(outContiguous, outPaged)
        }
    }

    // MARK: - Prefill: query-blocked and serialized

    /// `attendQueryBlocks` and the MTP serial path (`serializeQueries`,
    /// block size 1) both re-enter `attend` per block, so the coercion has
    /// to survive slicing. Oracle is the dtype-agnostic fp32 composed
    /// reference rather than the paged backend, which applies its own
    /// blocking policy at these lengths.
    @Test(arguments: [false, true])
    func blockedAndSerializedPrefillAcceptWiderSinks(serialize: Bool) throws {
        MLXRandom.seed(0x51_4B_D7_03)
        // Chunk widths come from the LIVE block width so each arm keeps
        // taking the route it is named for at any knob setting:
        //  - serialized: `serializeQueries` short-circuits the blocking gate
        //    entirely (`updateAndAttendRowSerialQueries` pins block size 1),
        //    so any multi-token chunk exercises it and the gate's verdict is
        //    irrelevant — asserting it here is what broke below 24;
        //  - unserialized: must EXCEED the live width to reach
        //    `attendQueryBlocks`, except under the `0` kill switch, where
        //    blocking is off for every L and the single-call terminal is
        //    what this arm covers instead.
        let chunk = serialize ? 24 : (liveBlock == 0 ? 200 : liveBlock + 24)
        let rowCapacity = max(512, chunk + 8)
        let kind = CBv2LayerKind(
            attention: .full, hasSinks: true, headDim: 64, kvHeads: 2, queryHeads: 8)
        let scale: Float = 0.125
        let (narrowSinks, wideSinks) = sinkPair(queryHeads: 8)
        if serialize {
            #expect(chunk > 1, "the serial path needs more than one query to split")
        } else {
            #expect(
                CBv2AttentionV1.shouldBlockQueries(chunk) == (liveBlock > 0),
                "chunk \(chunk) must block iff blocking is enabled (queryBlockSize \(liveBlock))")
        }

        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 29))
        var states: [[CBv2SequenceKV?]] = []
        defer { states.forEach { backend.release($0) } }
        func freshRow() throws -> CBv2SequenceKV {
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 0, maxLength: rowCapacity)
            states.append(state)
            return state[0]!
        }

        let q = MLXRandom.normal([1, 8, chunk, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)

        let wide = CBv2AttentionV1.updateAndAttend(
            rows: [try freshRow()], kind: kind, queries: q, keys: k, values: v,
            scale: scale, sinks: wideSinks, serializeQueries: serialize)
        let narrow = CBv2AttentionV1.updateAndAttend(
            rows: [try freshRow()], kind: kind, queries: q, keys: k, values: v,
            scale: scale, sinks: narrowSinks, serializeQueries: serialize)

        #expect(wide.shape == [1, 8, chunk, 64])
        assertIdentical(wide, narrow)
        assertClose(
            wide,
            PagedAttentionReference.composedAttention(
                queries: q, keys: k, values: v, scale: scale,
                boolMask: CBv2AttentionV1.boolMask(L: chunk, kL: chunk, window: nil),
                sinks: wideSinks))
    }

    // MARK: - Vision span chunks

    /// `attendSpanChunk` is the second MLXFast site and the only one reached
    /// with an explicit array mask. Span chunks never take the query-blocking
    /// path (the bidirectional overlay spans the whole chunk), so this is a
    /// single SDPA call with the widened sink.
    @Test func spanChunkAcceptsWiderSinks() throws {
        MLXRandom.seed(0x51_4B_D7_04)
        let chunk = 48
        let kind = CBv2LayerKind(
            attention: .full, hasSinks: true, headDim: 64, kvHeads: 2, queryHeads: 8)
        let scale: Float = 0.125
        let (narrowSinks, wideSinks) = sinkPair(queryHeads: 8)
        // Two bidirectional image blocks fully inside the chunk.
        let context = CBv2SpanChunkContext(
            chunkEnd: chunk,
            blocks: [
                CBv2ImageSpan(tokenOffset: 4, length: 12),
                CBv2ImageSpan(tokenOffset: 24, length: 16),
            ])

        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 29))
        var states: [[CBv2SequenceKV?]] = []
        defer { states.forEach { backend.release($0) } }
        func freshRow() throws -> CBv2SequenceKV {
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 0, maxLength: 512)
            states.append(state)
            return state[0]!
        }

        let q = MLXRandom.normal([1, 8, chunk, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)

        let wide = CBv2AttentionV1.updateAndAttend(
            rows: [try freshRow()], kind: kind, queries: q, keys: k, values: v,
            scale: scale, sinks: wideSinks, spanContexts: [context])
        let narrow = CBv2AttentionV1.updateAndAttend(
            rows: [try freshRow()], kind: kind, queries: q, keys: k, values: v,
            scale: scale, sinks: narrowSinks, spanContexts: [context])

        #expect(wide.shape == [1, 8, chunk, 64])
        assertIdentical(wide, narrow)
        assertClose(
            wide,
            PagedAttentionReference.composedAttention(
                queries: q, keys: k, values: v, scale: scale,
                boolMask: CBv2AttentionV1.spanChunkMask(
                    L: chunk, kL: chunk, window: nil, context: context),
                sinks: wideSinks))
    }

    // MARK: - The shared coercion contract

    /// `CBv2AttentionV1.sdpaSinks` is the single normalization every SDPA
    /// terminal calls, and the only part of this bug that can be asserted
    /// WITHOUT risking a process abort: every behavioural test above dies on
    /// a `fatalError` rather than a recorded issue when the coercion is
    /// missing, so this is the check that names the fault.
    @Test func sinkCoercionIsSharedContract() throws {
        let (narrow, wide) = sinkPair(queryHeads: 8)
        #expect(narrow.dtype == .float16)
        #expect(wide.dtype == .float32)

        // Wider-than-activation sinks are narrowed to the SDPA output dtype.
        #expect(CBv2AttentionV1.sdpaSinks(wide, queryDType: .float16)?.dtype == .float16)
        #expect(CBv2AttentionV1.sdpaSinks(wide, queryDType: .bfloat16)?.dtype == .bfloat16)
        // fp32 activations keep fp32 sinks: the cast is not unconditional.
        #expect(CBv2AttentionV1.sdpaSinks(wide, queryDType: .float32)?.dtype == .float32)
        // A sinkless layer stays sinkless.
        #expect(CBv2AttentionV1.sdpaSinks(nil, queryDType: .float16) == nil)

        // Matching dtypes must be a true no-op, not a rebuilt array: the
        // models shipping today load `sinks` in activation dtype and the
        // hoist is worthless if the common case still allocates.
        #expect(CBv2AttentionV1.sdpaSinks(narrow, queryDType: .float16) === narrow)
        #expect(CBv2AttentionV1.sdpaSinks(wide, queryDType: .float32) === wide)

        // Narrowing preserves the values (the fp32 arm is an fp16 widening),
        // so the two arms of every test above really are the same numbers.
        assertIdentical(
            try #require(CBv2AttentionV1.sdpaSinks(wide, queryDType: .float16)), narrow)
    }
}
