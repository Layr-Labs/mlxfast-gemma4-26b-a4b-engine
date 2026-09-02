// AttentionV1.swift
//
// The v1 attention dispatch used by `CBv2LayerCache.updateAndAttend`:
// per-row `MLXFast.scaledDotProductAttention` against each sequence's own
// contiguous KV. The paged backend (workstream C) replaces this behind the
// same `CBv2AttendingLayerCache` protocol.
//
// ## Why this path cannot have the left-padding bug class
// - Decode is rectangular [B, 1]: each row attends EXACTLY its own KV, with
//   no mask at all — a fully-masked row cannot exist by construction, so
//   NaN poisoning (ee2a921) is impossible.
// - Prefill is per-request [1, chunk]: masks are per-request, derived only
//   from that request's own lengths — batch composition cannot influence
//   them.
// - The mask mode is a PURE FUNCTION of (L, returned KV length, window):
//   never data-dependent, never drifting across steps for the same logical
//   computation (MLX #3384 / report 10 §4 invariant 5).

import Foundation
import MLX

enum CBv2AttentionV1 {

    private static let fusedRingWriteEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_FUSED_RING_WRITE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let queryBlockSize: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_ATTN_QUERY_BLOCK"],
            let value = Int(raw), value >= 0
        else { return 128 }
        return value
    }()

    static let wideHeadQueryBlockSize: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_ATTN_QUERY_BLOCK_WIDE"],
            let value = Int(raw), value >= 0
        else { return 128 }
        return value
    }()

    @inline(__always)
    private static func effectiveQueryBlockSize(
        kind: CBv2LayerKind, queryLength: Int
    ) -> Int {
        guard queryBlockSize == 128,
            wideHeadQueryBlockSize > 0,
            wideHeadQueryBlockSize < queryBlockSize,
            queryLength > queryBlockSize,
            kind.headDim == 256 || kind.headDim == 512
        else { return queryBlockSize }
        return wideHeadQueryBlockSize
    }

    static let packedBatchKVBudgetBytes: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_ATTN_BATCH_KV_BUDGET_MB"],
            let value = Int(raw), value >= 0
        else { return 512 << 20 }
        return value << 20
    }()

    static let packedKVAliasEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_PACKED_KV_ALIAS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let tokenMajorJoinEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_TOKENMAJOR_JOIN"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let joinKernelEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_JOIN_KERNEL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let maxJoinKernelBlocks = 30

    nonisolated(unsafe) private static var joinKernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let joinKernelLock = NSLock()

    private static func joinKernel(blockCount: Int) -> MLXFast.MLXFastKernel {
        joinKernelLock.lock()
        defer { joinKernelLock.unlock() }
        if let hit = joinKernels[blockCount] { return hit }
        let inputNames = (0 ..< blockCount).map { "in\($0)" }
        let cases = (0 ..< blockCount)
            .map { "                case \($0): src = in\($0); break;" }
            .joined(separator: "\n")
        let source = """
            const uint lanes = uint(D) / 8u;
            const uint L_ = uint(L);
            const uint H_ = uint(H);
            const uint BS_ = uint(BS);
            const uint D_ = uint(D);
            const uint dv = thread_position_in_grid.x;
            const uint h = thread_position_in_grid.y;
            const uint bt = thread_position_in_grid.z;
            if (dv >= lanes || h >= H_ || bt >= uint(B) * L_) {
                return;
            }
            const uint b = bt / L_;
            const uint t = bt - b * L_;
            const uint blk = t / BS_;
            const uint tt = t - blk * BS_;
            const device T* src;
            switch (blk) {
            \(cases)
                default: return;
            }
            const device uint4* s =
                (const device uint4*)(src + (((b * H_ + h) * BS_ + tt) * D_)) + dv;
            device uint4* o =
                (device uint4*)(out + (((b * L_ + t) * H_ + h) * D_)) + dv;
            *o = *s;
            """
        let kernel = MLXFast.metalKernel(
            name: "cbv2_prefill_join_v1_nb\(blockCount)",
            inputNames: inputNames,
            outputNames: ["out"],
            source: source,
            header: "#include <metal_stdlib>\nusing namespace metal;\n")
        joinKernels[blockCount] = kernel
        return kernel
    }

    private static func joinTokenMajor(_ blocks: [MLXArray]) -> MLXArray? {
        guard joinKernelEnabled, blocks.count >= 2, blocks.count <= maxJoinKernelBlocks
        else { return nil }
        let head = blocks[0]
        guard head.ndim == 4, head.dtype == .bfloat16 || head.dtype == .float16
        else { return nil }
        let (B, H, BS, D) = (head.dim(0), head.dim(1), head.dim(2), head.dim(3))
        guard B >= 1, H >= 1, BS >= 1, D >= 8, D % 8 == 0 else { return nil }
        for block in blocks {
            guard block.ndim == 4, block.dtype == head.dtype,
                block.dim(0) == B, block.dim(1) == H,
                block.dim(2) == BS, block.dim(3) == D
            else { return nil }
        }
        let L = BS * blocks.count
        let lanes = D / 8
        guard lanes <= 1024, B * L * H * D < (1 << 31) else { return nil }
        var headsPerGroup = 1
        for candidate in [16, 8, 4, 2]
        where H % candidate == 0 && lanes * candidate <= 1024 {
            headsPerGroup = candidate
            break
        }
        let joined = joinKernel(blockCount: blocks.count)(
            blocks.map { $0 as any ScalarOrArray },
            template: [
                ("T", head.dtype), ("L", L), ("BS", BS), ("H", H), ("D", D), ("B", B),
            ],
            grid: (lanes, H, B * L),
            threadGroup: (lanes, headsPerGroup, 1),
            outputShapes: [[B, L, H, D]],
            outputDTypes: [head.dtype]
        )[0]
        CBv2EngageMark.once("prefill-join-kernel")
        return joined
    }

    static let batchedFullDecodeEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_BATCHED_FULL_ATTENTION"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    @inline(__always)
    static func shouldBlockQueries(_ L: Int) -> Bool {
        queryBlockSize > 0 && L > queryBlockSize
    }

    @inline(__always)
    static func sdpaSinks(_ sinks: MLXArray?, queryDType: DType) -> MLXArray? {
        sinks?.asType(queryDType)
    }

    @inline(__always)
    private static func dispatchSinks(
        _ sinks: MLXArray?, kind: CBv2LayerKind, queries: MLXArray, softcap: Float?
    ) -> MLXArray? {
        guard kind.hasSinks, let sinks else { return nil }
        return softcap == nil ? sdpaSinks(sinks, queryDType: queries.dtype) : sinks
    }

    static func queryBlockBounds(
        historyCount: Int, offset: Int, count: Int, window: Int?
    ) -> (visibleStart: Int, visibleEnd: Int) {
        let visibleEnd = historyCount + offset + count
        let visibleStart = window.map { max(0, historyCount + offset + 1 - $0) } ?? 0
        return (visibleStart, visibleEnd)
    }

    static func maskMode(L: Int, kL: Int, window: Int?, bidirectional: Bool = false)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        if L == 1 { return .none }
        if bidirectional {
            return .array(boolMask(L: L, kL: kL, window: window, bidirectional: true)!)
        }
        if let window, kL > window {
            return .array(createCausalMask(n: L, offset: kL - L, windowSize: window))
        }
        return .causal
    }

    static func updateAndAttendColumns(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        columns: Int, scale: Float, sinks: MLXArray?, softcap: Float?,
        decodeRingWriteFence: CBv2DecodeRingWriteFence?,
        allowFusedRingWrite: Bool
    ) -> MLXArray? {
        let B = rows.count
        guard columns >= 2, B == 8, queries.dim(0) == B * columns,
            sinks == nil, softcap == nil,
            let decodeRingWriteFence, allowFusedRingWrite
        else { return nil }
        if case .full = kind.attention {
            return updateAndAttendFullColumns(
                rows: rows, kind: kind, queries: queries, keys: keys, values: values,
                columns: columns, scale: scale, decodeRingWriteFence: decodeRingWriteFence)
        }
        guard canUseRaggedTwoPassDecode(
                batch: B, cacheKind: kind, queryKind: kind,
                scale: scale, sinks: nil, softcap: nil),
            CBv2WindowedSequenceKV.q4FusedMirrorWriteEnabled,
            CBv2WindowedSequenceKV.q4BF16RingElideEnabled
        else { return nil }
        let ringRows = rows.compactMap { $0 as? CBv2WindowedSequenceKV }
        guard ringRows.count == B else { return nil }
        let preWrite = ringRows.compactMap { $0.decodeRingQuantViewBeforeWrite }
        guard preWrite.count == B else { return nil }
        let startArray =
            CBv2MTPWideVerifyContext.slidingStart(window: ringRows[0].window)
            ?? MLXArray(preWrite.map { UInt32($0.start) }, [B])
        guard let fused = CBv2RaggedTwoPassDecodeAttentionV1.attendRingQuantWritingColumns(
            queries: queries, mirrors: preWrite.map(\.mirror), startArray: startArray,
            newKeys: keys, newValues: values,
            previousWriteFence: decodeRingWriteFence.value, scale: scale,
            slidingWindowLength: ringRows[0].window, columns: columns)
        else { return nil }
        for row in ringRows {
            for _ in 0 ..< columns { row.advanceDecodeRingAfterQuantWrite() }
        }
        decodeRingWriteFence.value = fused.nextWriteFence
        return fused.output
    }

    private static func updateAndAttendFullColumns(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        columns: Int, scale: Float, decodeRingWriteFence: CBv2DecodeRingWriteFence
    ) -> MLXArray? {
        let fullRows = rows.compactMap { $0 as? CBv2FullSequenceKV }
        guard fullRows.count == rows.count,
            let maxOffset = fullRows.map(\.absoluteOffset).max(),
            let fused = CBv2RaggedComposedD512DecodeAttentionV1.updateAndAttendWritingRagged(
                rows: rows, kind: kind,
                queries: queries, keys: keys, values: values,
                previousWriteFence: decodeRingWriteFence.value,
                scale: scale, sinks: nil, softcap: nil,
                lengths: CBv2MTPWideVerifyContext.fullLengths()
                    ?? MLXArray(fullRows.map { Int32($0.absoluteOffset + 1) }),
                maxKeyLength: maxOffset + columns, columns: columns)
        else { return nil }
        decodeRingWriteFence.value = fused.nextWriteFence
        return fused.output
    }

    static func updateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float? = nil,
        spanContexts: [CBv2SpanChunkContext?]? = nil,
        serializeQueries: Bool = false,
        decodeRingWriteFence: CBv2DecodeRingWriteFence? = nil,
        allowFusedRingWrite: Bool = false,
        deviceColumn: Int = 0
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(!rows.isEmpty, "CBv2AttentionV1: no rows")
        precondition(
            B == rows.count,
            "CBv2AttentionV1: batch \(B) != bound cache rows \(rows.count)")
        if let spanContexts {
            precondition(
                spanContexts.count == B,
                "CBv2AttentionV1: span contexts \(spanContexts.count) != batch \(B)")
            precondition(
                L > 1,
                "CBv2AttentionV1: decode and one-token chunks never bind span masks")
            precondition(
                !serializeQueries || spanContexts.allSatisfy { $0 == nil },
                "CBv2AttentionV1: serialized MTP queries cannot carry span contexts")
        }
        let effectiveSinks = dispatchSinks(
            sinks, kind: kind, queries: queries, softcap: softcap)

        if serializeQueries, L > 1, B > 1 {
            var columns: [MLXArray] = []
            columns.reserveCapacity(L)
            for column in 0 ..< L {
                columns.append(
                    updateAndAttend(
                        rows: rows, kind: kind,
                        queries: queries[0..., 0..., column ..< (column + 1), 0...],
                        keys: keys[0..., 0..., column ..< (column + 1), 0...],
                        values: values[0..., 0..., column ..< (column + 1), 0...],
                        scale: scale, sinks: sinks, softcap: softcap,
                        spanContexts: nil,
                        serializeQueries: false,
                        decodeRingWriteFence: decodeRingWriteFence,
                        allowFusedRingWrite: allowFusedRingWrite,
                        deviceColumn: column))
            }
            CBv2EngageMark.once("mtp-wide-verify-attention-columns")
            return concatenated(columns, axis: 2)
        }

        if B == 1 {
            if serializeQueries, L > 1 {
                return updateAndAttendRowSerialQueries(
                    row: rows[0], kind: kind,
                    queries: queries, keys: keys, values: values,
                    scale: scale, sinks: effectiveSinks, softcap: softcap)
            }
            return updateAndAttendRow(
                row: rows[0], kind: kind,
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: effectiveSinks, softcap: softcap,
                spanContext: spanContexts?[0])
        }

        if L == 1 {
            if canUseRaggedTwoPassDecode(
                batch: B, cacheKind: kind, queryKind: kind,
                scale: scale, sinks: effectiveSinks, softcap: softcap)
            {
                var cachedKeyRows: [MLXArray] = []
                var cachedValueRows: [MLXArray] = []
                cachedKeyRows.reserveCapacity(B)
                cachedValueRows.reserveCapacity(B)
                let ringRows = rows.compactMap { $0 as? CBv2WindowedSequenceKV }
                if ringRows.count == B && ringRows.allSatisfy({ $0.decodeRingView != nil }) {
                    if CBv2WindowedSequenceKV.q4FusedMirrorWriteEnabled,
                        allowFusedRingWrite, let decodeRingWriteFence
                    {
                        let preWrite = ringRows.compactMap {
                            $0.decodeRingQuantViewBeforeWrite
                        }
                        let startArray =
                            CBv2MTPWideVerifyContext.slidingStart(
                                window: ringRows[0].window, column: deviceColumn)
                            ?? MLXArray(preWrite.map { UInt32($0.start) }, [B])
                        if preWrite.count == B,
                            let fused = CBv2RaggedTwoPassDecodeAttentionV1
                                .attendRingQuantWriting(
                                    queries: queries,
                                    mirrors: preWrite.map(\.mirror),
                                    startArray: startArray,
                                    newKeys: keys, newValues: values,
                                    previousWriteFence: decodeRingWriteFence.value,
                                    scale: scale,
                                    slidingWindowLength: ringRows[0].window)
                        {
                            if CBv2WindowedSequenceKV.q4BF16RingElideEnabled {
                                CBv2EngageMark.once("kvq4-bf16-elide")
                                for row in ringRows {
                                    row.advanceDecodeRingAfterQuantWrite()
                                }
                            } else {
                                for (index, row) in ringRows.enumerated() {
                                    row.decodeRingWriteBF16Only(
                                        keys: keys[index ..< (index + 1)],
                                        values: values[index ..< (index + 1)])
                                }
                            }
                            decodeRingWriteFence.value = fused.nextWriteFence
                            return fused.output
                        }
                    }
                    let portQuantActive = CBv2WindowedSequenceKV.quantEnabled
                        && ringRows.allSatisfy { $0.decodeRingQuantView != nil }
                    if !portQuantActive, fusedRingWriteEnabled, allowFusedRingWrite,
                        let decodeRingWriteFence
                    {
                        let preWrite = ringRows.compactMap { $0.decodeRingViewBeforeWrite }
                        if preWrite.count == B,
                            let fused = CBv2RaggedTwoPassDecodeAttentionV1
                                .attendRingWriting(
                                    queries: queries,
                                    newKeys: keys, newValues: values,
                                    keys: preWrite.map(\.keys),
                                    values: preWrite.map(\.values),
                                    starts: preWrite.map(\.start),
                                    previousWriteFence: decodeRingWriteFence.value,
                                    scale: scale,
                                    slidingWindowLength: ringRows[0].window)
                        {
                            for row in ringRows {
                                row.advanceDecodeRingAfterFusedWrite()
                            }
                            decodeRingWriteFence.value = fused.nextWriteFence
                            CBv2EngageMark.once("write016")
                            return fused.output
                        }
                    }

                    for (index, row) in ringRows.enumerated() {
                        row.decodeRingWrite(
                            keys: keys[index ..< (index + 1)],
                            values: values[index ..< (index + 1)])
                    }
                    let views = ringRows.compactMap { $0.decodeRingView }
                    let portMirrors = ringRows.compactMap { $0.decodeRingQuantView }
                    if views.count == B, portMirrors.count == B,
                        let quantOutput = CBv2RaggedTwoPassDecodeAttentionV1
                            .attendRingQuant(
                                queries: queries, mirrors: portMirrors,
                                starts: views.map(\.start), scale: scale,
                                slidingWindowLength: ringRows[0].window)
                    {
                        return quantOutput
                    }
                    if views.count == B, !ringRows.contains(where: \.bf16RingStale),
                        let output = CBv2RaggedTwoPassDecodeAttentionV1.attendRing(
                            queries: queries, keys: views.map(\.keys),
                            values: views.map(\.values), starts: views.map(\.start),
                            scale: scale, slidingWindowLength: ringRows[0].window)
                    {
                        return output
                    }
                    for row in ringRows {
                        let view = row.snapshot()
                        cachedKeyRows.append(view.keys)
                        cachedValueRows.append(view.values)
                    }
                } else {
                    for (index, row) in rows.enumerated() {
                        let (cachedKeys, cachedValues) = row.update(
                            keys: keys[index ..< (index + 1)],
                            values: values[index ..< (index + 1)])
                        cachedKeyRows.append(cachedKeys)
                        cachedValueRows.append(cachedValues)
                    }
                }
                if let output = CBv2RaggedTwoPassDecodeAttentionV1.attend(
                    queries: queries, keys: cachedKeyRows, values: cachedValueRows,
                    scale: scale)
                {
                    return output
                }

                var outputs: [MLXArray] = []
                outputs.reserveCapacity(B)
                for index in 0 ..< B {
                    outputs.append(
                        attend(
                            queries: queries[index ..< (index + 1)],
                            keys: cachedKeyRows[index], values: cachedValueRows[index],
                            scale: scale, L: 1, kL: cachedKeyRows[index].dim(2),
                            window: nil, sinks: effectiveSinks, softcap: softcap))
                }
                return concatenated(outputs, axis: 0)
            }

            if let decodeRingWriteFence, allowFusedRingWrite,
                case .full = kind.attention,
                let fullRows = Optional(rows.compactMap { $0 as? CBv2FullSequenceKV }),
                fullRows.count == B,
                let first = fullRows.first?.absoluteOffset,
                CBv2MTPWideVerifyContext.deviceBase != nil
                    || !fullRows.allSatisfy({ $0.absoluteOffset == first }),
                let fused = CBv2RaggedComposedD512DecodeAttentionV1
                    .updateAndAttendWritingRagged(
                        rows: rows, kind: kind,
                        queries: queries, keys: keys, values: values,
                        previousWriteFence: decodeRingWriteFence.value,
                        scale: scale, sinks: effectiveSinks, softcap: softcap,
                        lengths: CBv2MTPWideVerifyContext.fullLengths(column: deviceColumn)
                            ?? MLXArray(fullRows.map { Int32($0.absoluteOffset + 1) }),
                        maxKeyLength: fullRows.map { $0.absoluteOffset + 1 }.max()!)
            {
                decodeRingWriteFence.value = fused.nextWriteFence
                return fused.output
            }

            if let decodeRingWriteFence, allowFusedRingWrite,
                let fused = CBv2RaggedComposedD512DecodeAttentionV1
                    .updateAndAttendWriting22(
                        rows: rows, kind: kind,
                        queries: queries, keys: keys, values: values,
                        previousWriteFence: decodeRingWriteFence.value,
                        scale: scale, sinks: effectiveSinks, softcap: softcap)
            {
                decodeRingWriteFence.value = fused.nextWriteFence
                CBv2EngageMark.once("write022d512")
                return fused.output
            }

            if let decodeRingWriteFence, allowFusedRingWrite,
                let fused = CBv2RaggedComposedD512DecodeAttentionV1
                    .updateAndAttendWriting(
                        rows: rows, kind: kind,
                        queries: queries, keys: keys, values: values,
                        previousWriteFence: decodeRingWriteFence.value,
                        scale: scale, sinks: effectiveSinks, softcap: softcap)
            {
                decodeRingWriteFence.value = fused.nextWriteFence
                CBv2EngageMark.once("write016d512")
                return fused.output
            }

            if let output = CBv2RaggedComposedD512DecodeAttentionV1.updateAndAttend(
                rows: rows, kind: kind,
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: effectiveSinks, softcap: softcap)
            {
                CBv2EngageMark.once("d512sdpa")
                return output
            }

            if let output = batchedFullDecodeUpdateAndAttend(
                rows: rows, kind: kind,
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: effectiveSinks, softcap: softcap)
            {
                CBv2EngageMark.once("att008")
                return output
            }

            var outputs: [MLXArray] = []
            outputs.reserveCapacity(B)
            for (index, row) in rows.enumerated() {
                let (cachedKeys, cachedValues) = row.update(
                    keys: keys[index ..< (index + 1)],
                    values: values[index ..< (index + 1)])
                outputs.append(
                    attend(
                        queries: queries[index ..< (index + 1)],
                        keys: cachedKeys, values: cachedValues, scale: scale,
                        L: 1, kL: cachedKeys.dim(2), window: nil,
                        sinks: effectiveSinks, softcap: softcap))
            }
            return concatenated(outputs, axis: 0)
        }

        if serializeQueries {
            return packedPerRow(batch: B) { index, slice in
                updateAndAttendRowSerialQueries(
                    row: rows[index], kind: kind,
                    queries: slice(queries), keys: slice(keys), values: slice(values),
                    scale: scale, sinks: effectiveSinks, softcap: softcap)
            }
        }

        var cachedKeys: [MLXArray] = []
        var cachedValues: [MLXArray] = []
        cachedKeys.reserveCapacity(B)
        cachedValues.reserveCapacity(B)
        let windowedRows = rows.compactMap { $0 as? CBv2WindowedSequenceKV }
        if allowFusedRingWrite, let decodeRingWriteFence, windowedRows.count == B,
            let written = CBv2WindowedSequenceKV.batchedChunkUpdate(
                rows: windowedRows, keys: keys, values: values,
                previousWriteFence: decodeRingWriteFence.value)
        {
            decodeRingWriteFence.value = written.nextWriteFence
            cachedKeys = written.views.map(\.keys)
            cachedValues = written.views.map(\.values)
            CBv2EngageMark.once("prefill-kv-batched-write")
        } else {
            for (index, row) in rows.enumerated() {
                let (rowKeys, rowValues) = row.update(
                    keys: keys[index ..< (index + 1)],
                    values: values[index ..< (index + 1)])
                cachedKeys.append(rowKeys)
                cachedValues.append(rowValues)
            }
        }

        if let batched = batchedPackedAttention(
            kind: kind, queries: queries,
            cachedKeys: cachedKeys, cachedValues: cachedValues,
            window: window(of: kind), scale: scale,
            sinks: effectiveSinks, softcap: softcap, spanContexts: spanContexts,
            newKeys: keys, newValues: values)
        {
            return batched
        }

        return packedPerRow(batch: B) { index, slice in
            attendCommittedRow(
                kind: kind, queries: slice(queries),
                cachedKeys: cachedKeys[index], cachedValues: cachedValues[index],
                window: window(of: kind), scale: scale,
                sinks: effectiveSinks, softcap: softcap,
                spanContext: spanContexts?[index])
        }
    }

    private static func batchedPackedAttention(
        kind: CBv2LayerKind, queries: MLXArray,
        cachedKeys: [MLXArray], cachedValues: [MLXArray],
        window: Int?, scale: Float, sinks: MLXArray?, softcap: Float?,
        spanContexts: [CBv2SpanChunkContext?]?,
        newKeys: MLXArray? = nil, newValues: MLXArray? = nil
    ) -> MLXArray? {
        guard packedBatchKVBudgetBytes > 0 else { return nil }
        guard spanContexts?.allSatisfy({ $0 == nil }) ?? true else { return nil }
        guard let headKeys = cachedKeys.first, let headValues = cachedValues.first,
            cachedKeys.count == cachedValues.count, cachedKeys.count > 1
        else { return nil }
        guard headKeys.ndim == 4, headValues.ndim == 4 else { return nil }
        let kL = headKeys.dim(2)
        guard headValues.dim(2) == kL else { return nil }
        for index in cachedKeys.indices {
            let rowKeys = cachedKeys[index]
            let rowValues = cachedValues[index]
            guard rowKeys.ndim == 4, rowValues.ndim == 4,
                rowKeys.dim(0) == 1, rowValues.dim(0) == 1,
                rowKeys.dim(2) == kL, rowValues.dim(2) == kL,
                rowKeys.dim(1) == headKeys.dim(1), rowKeys.dim(3) == headKeys.dim(3),
                rowValues.dim(1) == headValues.dim(1),
                rowValues.dim(3) == headValues.dim(3),
                rowKeys.dtype == headKeys.dtype, rowValues.dtype == headValues.dtype
            else { return nil }
        }
        let restackedBytes =
            (headKeys.nbytes + headValues.nbytes) * cachedKeys.count
        guard restackedBytes <= packedBatchKVBudgetBytes else { return nil }

        if packedKVAliasEnabled,
            let newKeys, let newValues,
            newKeys.ndim == 4, newValues.ndim == 4,
            newKeys.dim(2) == kL, newValues.dim(2) == kL,
            newKeys.dim(0) == cachedKeys.count,
            newValues.dim(0) == cachedKeys.count,
            newKeys.dim(1) == headKeys.dim(1),
            newKeys.dim(3) == headKeys.dim(3),
            newValues.dim(1) == headValues.dim(1),
            newValues.dim(3) == headValues.dim(3),
            newKeys.dtype == headKeys.dtype,
            newValues.dtype == headValues.dtype
        {
            CBv2EngageMark.once("prefill-packedkv-alias")
            return attendCommittedRow(
                kind: kind, queries: queries,
                cachedKeys: contiguous(newKeys),
                cachedValues: contiguous(newValues),
                window: window, scale: scale, sinks: sinks, softcap: softcap,
                spanContext: nil)
        }

        return attendCommittedRow(
            kind: kind, queries: queries,
            cachedKeys: concatenated(cachedKeys, axis: 0),
            cachedValues: concatenated(cachedValues, axis: 0),
            window: window, scale: scale, sinks: sinks, softcap: softcap,
            spanContext: nil)
    }

    private static func attendCommittedRow(
        kind: CBv2LayerKind, queries: MLXArray,
        cachedKeys: MLXArray, cachedValues: MLXArray,
        window: Int?, scale: Float, sinks: MLXArray?, softcap: Float?,
        spanContext: CBv2SpanChunkContext?
    ) -> MLXArray {
        let L = queries.dim(2)
        let blockSize = effectiveQueryBlockSize(kind: kind, queryLength: L)
        if blockSize > 0 && L > blockSize && !kind.isBidirectional {
            return attendQueryBlocks(
                queries: queries, keys: cachedKeys, values: cachedValues,
                newTokenCount: L, window: window, scale: scale,
                sinks: sinks, softcap: softcap, blockSize: blockSize,
                spanContext: spanContext)
        }
        if let spanContext {
            return attendSpanChunk(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                L: L, kL: cachedKeys.dim(2), window: window,
                context: spanContext, sinks: sinks, softcap: softcap)
        }
        return attend(
            queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
            L: L, kL: cachedKeys.dim(2), window: window,
            sinks: sinks, softcap: softcap, bidirectional: kind.isBidirectional)
    }

    @inline(__always)
    static func packedPerRow(
        batch: Int, _ row: (_ index: Int, _ slice: (MLXArray) -> MLXArray) -> MLXArray
    ) -> MLXArray {
        precondition(batch >= 1, "CBv2AttentionV1: packed batch must be >= 1")
        if batch == 1 { return row(0, { $0 }) }
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(batch)
        for index in 0 ..< batch {
            outputs.append(row(index, { $0[index ..< (index + 1)] }))
        }
        return concatenated(outputs, axis: 0)
    }

    static func updateAndAttendLastQuery(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float? = nil
    ) -> MLXArray {
        precondition(
            kind.attention == .full,
            "CBv2AttentionV1: last-query prefill requires full attention")
        precondition(
            kind.sharesKVWithLayer == nil,
            "CBv2AttentionV1: last-query prefill cannot own KV for a shared layer")
        precondition(
            queries.ndim == 4 && keys.ndim == 4 && values.ndim == 4,
            "CBv2AttentionV1: last-query prefill tensors must be rank 4")
        let batch = queries.dim(0)
        precondition(
            batch > 0 && rows.count == batch
                && keys.dim(0) == batch && values.dim(0) == batch,
            "CBv2AttentionV1: last-query prefill rows and tensors must share batch B")
        precondition(
            queries.dim(2) == 1,
            "CBv2AttentionV1: last-query prefill requires qL=1")
        let kvLength = keys.dim(2)
        precondition(kvLength > 1, "CBv2AttentionV1: last-query prefill requires kvL>1")
        precondition(
            values.dim(2) == kvLength,
            "CBv2AttentionV1: last-query prefill K/V lengths must match")
        precondition(
            queries.dim(1) == kind.queryHeads && queries.dim(3) == kind.headDim,
            "CBv2AttentionV1: last-query prefill Q shape does not match the layer kind")
        precondition(
            keys.dim(1) == kind.kvHeads && values.dim(1) == kind.kvHeads
                && keys.dim(3) == kind.headDim && values.dim(3) == kind.headDim,
            "CBv2AttentionV1: last-query prefill K/V shape does not match the layer kind")

        let effectiveSinks = dispatchSinks(
            sinks, kind: kind, queries: queries, softcap: softcap)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(batch)
        for (index, row) in rows.enumerated() {
            let (cachedKeys, cachedValues) = row.update(
                keys: keys[index ..< index + 1],
                values: values[index ..< index + 1])
            outputs.append(
                attend(
                    queries: queries[index ..< index + 1],
                    keys: cachedKeys, values: cachedValues,
                    scale: scale, L: 1, kL: cachedKeys.dim(2), window: nil,
                    sinks: effectiveSinks, softcap: softcap))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    private static func updateAndAttendRow(
        row: CBv2SequenceKV, kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?,
        spanContext: CBv2SpanChunkContext?
    ) -> MLXArray {
        let (cachedKeys, cachedValues) = row.update(keys: keys, values: values)
        return attendCommittedRow(
            kind: kind, queries: queries,
            cachedKeys: cachedKeys, cachedValues: cachedValues,
            window: window(of: kind), scale: scale,
            sinks: sinks, softcap: softcap, spanContext: spanContext)
    }

    private static func updateAndAttendRowSerialQueries(
        row: CBv2SequenceKV, kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        let L = queries.dim(2)
        let (cachedKeys, cachedValues) = row.update(keys: keys, values: values)
        return attendSerialQueries(
            queries: queries, keys: cachedKeys, values: cachedValues,
            newTokenCount: L, window: window(of: kind), scale: scale,
            sinks: sinks, softcap: softcap)
    }

    static func attendBorrowing(
        sourceRows: [CBv2SequenceKV], sourceKind: CBv2LayerKind, kind: CBv2LayerKind,
        queries: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float? = nil,
        spanContexts: [CBv2SpanChunkContext?]? = nil,
        serializeQueries: Bool = false
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(!sourceRows.isEmpty, "CBv2AttentionV1: no source rows to borrow from")
        precondition(
            B == sourceRows.count,
            "CBv2AttentionV1: batch \(B) != source rows \(sourceRows.count)")
        if let spanContexts {
            precondition(
                spanContexts.count == B,
                "CBv2AttentionV1: span contexts \(spanContexts.count) != batch \(B)")
            precondition(
                L > 1,
                "CBv2AttentionV1: decode and one-token chunks never bind span masks")
            precondition(
                !serializeQueries || spanContexts.allSatisfy { $0 == nil },
                "CBv2AttentionV1: serialized MTP queries cannot carry span contexts")
        }
        let effectiveSinks = dispatchSinks(
            sinks, kind: kind, queries: queries, softcap: softcap)

        if L > 1 {
            if B == 1 {
                if serializeQueries {
                    let (keys, values) = chunkBorrowViews(of: sourceRows[0])
                    return attendSerialQueries(
                        queries: queries, keys: keys, values: values,
                        newTokenCount: L, window: window(of: sourceKind), scale: scale,
                        sinks: effectiveSinks, softcap: softcap)
                }
                return borrowAndAttendRow(
                    sourceRow: sourceRows[0], sourceKind: sourceKind,
                    queries: queries, scale: scale, sinks: effectiveSinks, softcap: softcap,
                    spanContext: spanContexts?[0])
            }
            if !serializeQueries {
                var borrowedKeys: [MLXArray] = []
                var borrowedValues: [MLXArray] = []
                borrowedKeys.reserveCapacity(B)
                borrowedValues.reserveCapacity(B)
                for row in sourceRows {
                    let (rowKeys, rowValues) = chunkBorrowViews(of: row)
                    borrowedKeys.append(rowKeys)
                    borrowedValues.append(rowValues)
                }
                if let batched = batchedPackedAttention(
                    kind: sourceKind, queries: queries,
                    cachedKeys: borrowedKeys, cachedValues: borrowedValues,
                    window: window(of: sourceKind), scale: scale,
                    sinks: effectiveSinks, softcap: softcap,
                    spanContexts: spanContexts)
                {
                    return batched
                }
            }
            var outputs: [MLXArray] = []
            outputs.reserveCapacity(B)
            for (index, row) in sourceRows.enumerated() {
                if serializeQueries {
                    let (keys, values) = chunkBorrowViews(of: row)
                    outputs.append(
                        attendSerialQueries(
                            queries: queries[index ..< (index + 1)],
                            keys: keys, values: values, newTokenCount: L,
                            window: window(of: sourceKind), scale: scale,
                            sinks: effectiveSinks, softcap: softcap))
                } else {
                    outputs.append(
                        borrowAndAttendRow(
                        sourceRow: row, sourceKind: sourceKind,
                        queries: queries[index ..< (index + 1)],
                        scale: scale, sinks: effectiveSinks, softcap: softcap,
                        spanContext: spanContexts?[index]))
                }
            }
            return concatenated(outputs, axis: 0)
        }

        if canUseRaggedTwoPassDecode(
            batch: B, cacheKind: sourceKind, queryKind: kind,
            scale: scale, sinks: effectiveSinks, softcap: softcap)
        {
            var cachedKeyRows: [MLXArray] = []
            var cachedValueRows: [MLXArray] = []
            cachedKeyRows.reserveCapacity(B)
            cachedValueRows.reserveCapacity(B)
            for row in sourceRows {
                let cachedKeys: MLXArray
                let cachedValues: MLXArray
                if let windowed = row as? CBv2WindowedSequenceKV {
                    (cachedKeys, cachedValues) = windowed.decodeBorrowableViews()
                } else {
                    (cachedKeys, cachedValues, _) = row.snapshot()
                }
                cachedKeyRows.append(cachedKeys)
                cachedValueRows.append(cachedValues)
            }
            if let output = CBv2RaggedTwoPassDecodeAttentionV1.attend(
                queries: queries, keys: cachedKeyRows, values: cachedValueRows,
                scale: scale)
            {
                return output
            }

            var outputs: [MLXArray] = []
            outputs.reserveCapacity(B)
            for index in 0 ..< B {
                outputs.append(
                    attend(
                        queries: queries[index ..< (index + 1)],
                        keys: cachedKeyRows[index], values: cachedValueRows[index],
                        scale: scale, L: 1, kL: cachedKeyRows[index].dim(2), window: nil,
                        sinks: effectiveSinks, softcap: softcap))
            }
            return concatenated(outputs, axis: 0)
        }

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (index, row) in sourceRows.enumerated() {
            let cachedKeys: MLXArray
            let cachedValues: MLXArray
            if let windowed = row as? CBv2WindowedSequenceKV {
                (cachedKeys, cachedValues) = windowed.decodeBorrowableViews()
            } else {
                (cachedKeys, cachedValues, _) = row.snapshot()
            }
            outputs.append(
                attend(
                    queries: B == 1 ? queries : queries[index ..< (index + 1)],
                    keys: cachedKeys, values: cachedValues, scale: scale,
                    L: 1, kL: cachedKeys.dim(2), window: nil,
                    sinks: effectiveSinks, softcap: softcap))
        }
        return B == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    private static func borrowAndAttendRow(
        sourceRow: CBv2SequenceKV, sourceKind: CBv2LayerKind,
        queries: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float?,
        spanContext: CBv2SpanChunkContext?
    ) -> MLXArray {
        let L = queries.dim(2)
        let (cachedKeys, cachedValues) = chunkBorrowViews(of: sourceRow)
        let blockSize = effectiveQueryBlockSize(kind: sourceKind, queryLength: L)
        if blockSize > 0 && L > blockSize && !sourceKind.isBidirectional {
            return attendQueryBlocks(
                queries: queries, keys: cachedKeys, values: cachedValues,
                newTokenCount: L, window: window(of: sourceKind), scale: scale,
                sinks: sinks, softcap: softcap, blockSize: blockSize,
                spanContext: spanContext)
        }
        if let spanContext {
            return attendSpanChunk(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                L: L, kL: cachedKeys.dim(2), window: window(of: sourceKind),
                context: spanContext, sinks: sinks, softcap: softcap)
        }
        return attend(
            queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
            L: L, kL: cachedKeys.dim(2), window: window(of: sourceKind),
            sinks: sinks, softcap: softcap, bidirectional: sourceKind.isBidirectional)
    }

    private static func chunkBorrowViews(of row: CBv2SequenceKV) -> (MLXArray, MLXArray) {
        if let windowed = row as? CBv2WindowedSequenceKV {
            return windowed.borrowableViews()
        }
        let (keys, values, _) = row.snapshot()
        return (keys, values)
    }

    // MARK: - Private

    private static func batchedFullDecodeUpdateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray? {
        guard batchedFullDecodeEnabled,
            rows.count == 8,
            scale == 1.0,
            sinks == nil,
            softcap == nil,
            !kind.isBidirectional,
            kind.kvHeads == 2,
            kind.headDim == 512,
            kind.queryHeads == 16,
            queries.dtype == .bfloat16,
            keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            queries.shape == [8, 16, 1, 512],
            keys.shape == [8, 2, 1, 512],
            values.shape == [8, 2, 1, 512]
        else { return nil }
        guard case .full = kind.attention else { return nil }

        let fullRows = rows.compactMap { $0 as? CBv2FullSequenceKV }
        guard fullRows.count == 8 else { return nil }

        let offset = fullRows[0].absoluteOffset
        guard offset > 0,
            fullRows.allSatisfy({ $0.absoluteOffset == offset }),
            fullRows.allSatisfy({ offset + 1 <= $0.maxLength })
        else { return nil }

        guard let pool = CBv2FullSequenceKV.cohortPool(binding: fullRows) else {
            return nil
        }

        pool.batchAppend(keys: keys, values: values, at: offset)
        for row in fullRows {
            row.confirmPooledBatchAppend(1)
        }

        let (cachedKeys, cachedValues) = pool.batchViews(upTo: offset + 1)
        return attend(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, L: 1, kL: offset + 1, window: nil,
            sinks: nil, softcap: nil)
    }

    @inline(__always)
    private static func canUseRaggedTwoPassDecode(
        batch: Int, cacheKind: CBv2LayerKind, queryKind: CBv2LayerKind,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> Bool {
        guard batch == 8,
            scale == 1.0,
            sinks == nil,
            softcap == nil,
            !cacheKind.isBidirectional,
            !queryKind.isBidirectional,
            cacheKind.kvHeads == 8,
            cacheKind.headDim == 256,
            queryKind.queryHeads == 16,
            queryKind.headDim == 256
        else { return false }

        guard case .slidingWindow(let window) = cacheKind.attention else {
            return false
        }
        return window == 1024
    }

    private static func window(of kind: CBv2LayerKind) -> Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    private static func attendQueryBlocks(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        newTokenCount: Int, window: Int?, scale: Float,
        sinks: MLXArray?, softcap: Float?, blockSize: Int,
        spanContext: CBv2SpanChunkContext? = nil
    ) -> MLXArray {
        precondition(blockSize >= 1, "CBv2AttentionV1: query block size must be >= 1")
        let keyCount = keys.dim(2)
        let historyCount = keyCount - newTokenCount
        precondition(historyCount >= 0)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity((newTokenCount + blockSize - 1) / blockSize)
        let queryPlane: MLXArray? =
            (blockSize > 8 && newTokenCount > 8)
            ? CBv2ComposedPrefillSDPAV1.queryPlane(
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: sinks, softcap: softcap)
            : nil
        var offset = 0
        while offset < newTokenCount {
            let count = min(blockSize, newTokenCount - offset)
            let bounds = queryBlockBounds(
                historyCount: historyCount, offset: offset, count: count, window: window)
            var visibleStart = bounds.visibleStart
            var visibleEnd = bounds.visibleEnd
            var intersectingBlocks: ArraySlice<CBv2ImageSpan>?
            let queryAbsoluteStart =
                spanContext.map { $0.chunkEnd - newTokenCount + offset }
            let keyAbsoluteStart = spanContext.map { $0.chunkEnd - keyCount }
            if let context = spanContext,
                let qStart = queryAbsoluteStart,
                let kStart = keyAbsoluteStart
            {
                let blocks = spanBlocksIntersectingQueryRange(
                    queryAbsoluteStart: qStart, queryCount: count, context: context)
                intersectingBlocks = blocks
                for block in blocks {
                    visibleStart = min(visibleStart, max(0, block.tokenOffset - kStart))
                    visibleEnd = max(visibleEnd, min(keyCount, block.end - kStart))
                }
            }

            let querySlice = queries[0..., 0..., offset ..< (offset + count), 0...]
            let keySlice = keys[0..., 0..., visibleStart ..< visibleEnd, 0...]
            let valueSlice = values[0..., 0..., visibleStart ..< visibleEnd, 0...]
            if let blocks = intersectingBlocks, !blocks.isEmpty,
                let qStart = queryAbsoluteStart,
                let kStart = keyAbsoluteStart
            {
                outputs.append(
                    attendSpanSlice(
                        queries: querySlice, keys: keySlice, values: valueSlice,
                        scale: scale, queryAbsoluteStart: qStart,
                        keyAbsoluteStart: kStart + visibleStart,
                        window: window, blocks: blocks,
                        sinks: sinks, softcap: softcap))
            } else {
                let planeSlice = queryPlane.map {
                    $0[0..., 0..., 0..., offset ..< (offset + count), 0...]
                }
                outputs.append(
                    attend(
                        queries: querySlice, keys: keySlice, values: valueSlice,
                        scale: scale, L: count, kL: visibleEnd - visibleStart,
                        window: window, sinks: sinks, softcap: softcap,
                        queryPlaneSlice: planeSlice))
            }
            offset += count
        }
        if outputs.count == 1 { return outputs[0] }
        if blockSize > 8, newTokenCount > 8,
            let joined = joinTokenMajor(outputs)
        {
            return joined.transposed(0, 2, 1, 3)
        }
        if tokenMajorJoinEnabled,
            blockSize > 8,
            newTokenCount > 8,
            outputs[0].ndim == 4
        {
            CBv2EngageMark.once("prefill-tokenmajor-join")
            let tokenMajor = concatenated(
                outputs.map { $0.transposed(0, 2, 1, 3) }, axis: 1)
            return tokenMajor.transposed(0, 2, 1, 3)
        }
        return concatenated(outputs, axis: 2)
    }

    private static func attendSerialQueries(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        newTokenCount: Int, window: Int?, scale: Float,
        sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        attendQueryBlocks(
            queries: queries, keys: keys, values: values,
            newTokenCount: newTokenCount, window: window, scale: scale,
            sinks: sinks, softcap: softcap, blockSize: 1)
    }

    private static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        L: Int, kL: Int, window: Int?, sinks: MLXArray?, softcap: Float?,
        bidirectional: Bool = false, queryPlaneSlice: MLXArray? = nil
    ) -> MLXArray {
        let attentionKeys = keys.dtype == queries.dtype ? keys : keys.asType(queries.dtype)
        let attentionValues =
            values.dtype == queries.dtype ? values : values.asType(queries.dtype)
        guard let softcap else {
            assert(
                sinks == nil || sinks!.dtype == queries.dtype,
                "CBv2AttentionV1: sinks must be normalized to the query dtype before SDPA")
            if let composed = CBv2ComposedPrefillSDPAV1.attend(
                queries: queries, keys: attentionKeys, values: attentionValues,
                scale: scale, L: L, kL: kL, window: window,
                bidirectional: bidirectional, sinks: sinks,
                queryPlaneSlice: queryPlaneSlice)
            {
                return composed
            }
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
                mask: maskMode(
                    L: L, kL: kL, window: window, bidirectional: bidirectional),
                sinks: sinks)
        }
        return PagedAttentionReference.composedAttention(
            queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
            boolMask: boolMask(
                L: L, kL: kL, window: window, bidirectional: bidirectional),
            sinks: sinks, softcap: softcap)
    }

    static func boolMask(
        L: Int, kL: Int, window: Int?, bidirectional: Bool = false
    ) -> MLXArray? {
        guard L > 1 else { return nil }
        let qpos = MLXArray(Int32(kL - L) ..< Int32(kL)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(kL)).expandedDimensions(axis: 0)
        var mask = kpos .<= qpos
        if let window, kL > window {
            mask = mask .&& (kpos .> (qpos - Int32(window)))
        }
        if bidirectional {
            var reverse = (kpos .>= qpos) .&& (kpos .>= Int32(kL - L))
            if let window {
                reverse = reverse .&& (kpos .< (qpos + Int32(window)))
            }
            mask = mask .|| reverse
        }
        return mask
    }

    // MARK: - Vision span-chunk path (pinned; span-containing chunks only)

    static func spanBlocksIntersectingQueryRange(
        queryAbsoluteStart: Int, queryCount: Int,
        context: CBv2SpanChunkContext
    ) -> ArraySlice<CBv2ImageSpan> {
        precondition(queryCount >= 0)
        let queryEnd = queryAbsoluteStart + queryCount
        var lower = context.blocks.startIndex
        while lower < context.blocks.endIndex,
            context.blocks[lower].end <= queryAbsoluteStart
        {
            lower += 1
        }
        var upper = lower
        while upper < context.blocks.endIndex,
            context.blocks[upper].tokenOffset < queryEnd
        {
            upper += 1
        }
        return context.blocks[lower ..< upper]
    }

    private static func spanMask(
        queryAbsoluteStart: Int, queryCount: Int,
        keyAbsoluteStart: Int, keyCount: Int,
        window: Int?, blocks: ArraySlice<CBv2ImageSpan>
    ) -> MLXArray {
        let qAbs =
            MLXArray(
                Int32(queryAbsoluteStart) ..< Int32(queryAbsoluteStart + queryCount)
            ).expandedDimensions(axis: 1)
        let kAbs =
            MLXArray(
                Int32(keyAbsoluteStart) ..< Int32(keyAbsoluteStart + keyCount)
            ).expandedDimensions(axis: 0)
        var mask = kAbs .<= qAbs
        if let window {
            mask = mask .&& (kAbs .> (qAbs - Int32(window)))
        }
        for block in blocks {
            let lo = Int32(block.tokenOffset)
            let hi = Int32(block.end)
            let qIn = (qAbs .>= lo) .&& (qAbs .< hi)
            let kIn = (kAbs .>= lo) .&& (kAbs .< hi)
            mask = mask .|| (qIn .&& kIn)
        }
        return mask
    }

    static func spanChunkMask(
        L: Int, kL: Int, window: Int?, context: CBv2SpanChunkContext
    ) -> MLXArray {
        precondition(L > 1, "span chunks are multi-token by construction")
        return spanMask(
            queryAbsoluteStart: context.chunkEnd - L, queryCount: L,
            keyAbsoluteStart: context.chunkEnd - kL, keyCount: kL,
            window: window, blocks: context.blocks[...])
    }

    private static func attendSpanChunk(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        L: Int, kL: Int, window: Int?, context: CBv2SpanChunkContext,
        sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        attendSpanSlice(
            queries: queries, keys: keys, values: values, scale: scale,
            queryAbsoluteStart: context.chunkEnd - L,
            keyAbsoluteStart: context.chunkEnd - kL,
            window: window, blocks: context.blocks[...],
            sinks: sinks, softcap: softcap)
    }

    private static func attendSpanSlice(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        queryAbsoluteStart: Int, keyAbsoluteStart: Int,
        window: Int?, blocks: ArraySlice<CBv2ImageSpan>,
        sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        let attentionKeys = keys.dtype == queries.dtype ? keys : keys.asType(queries.dtype)
        let attentionValues =
            values.dtype == queries.dtype ? values : values.asType(queries.dtype)
        let mask = spanMask(
            queryAbsoluteStart: queryAbsoluteStart, queryCount: queries.dim(2),
            keyAbsoluteStart: keyAbsoluteStart, keyCount: keys.dim(2),
            window: window, blocks: blocks)
        guard let softcap else {
            assert(
                sinks == nil || sinks!.dtype == queries.dtype,
                "CBv2AttentionV1: sinks must be normalized to the query dtype before SDPA")
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
                mask: .array(mask), sinks: sinks)
        }
        return PagedAttentionReference.composedAttention(
            queries: queries, keys: attentionKeys, values: attentionValues, scale: scale,
            boolMask: mask, sinks: sinks, softcap: softcap)
    }
}
