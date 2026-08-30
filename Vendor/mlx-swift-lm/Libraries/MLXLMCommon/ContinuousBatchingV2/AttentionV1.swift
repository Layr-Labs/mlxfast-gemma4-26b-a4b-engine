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

/// Namespace for the v1 (per-row SDPA) attention dispatch.
enum CBv2AttentionV1 {

    /// Kill switch for the fused full-ring decode write (see
    /// `CBv2RaggedTwoPassDecodeAttentionV1.attendRingWriting`).
    /// `0`/`false`/`no`/`off` restores the established `decodeRingWrite` +
    /// `attendRing` pair, which also stays the fallback for every input the
    /// fused path refuses.
    private static let fusedRingWriteEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_FUSED_RING_WRITE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Query-block width for multi-token prompt attention (see
    /// `attendQueryBlocks`). Smaller blocks execute strictly less attention
    /// work and hold a smaller score tensor, but cost one dispatch set each;
    /// the composed path is ~4-5 Metal dispatches per call, so very small
    /// blocks trade GPU efficiency for FLOPs already saved.
    ///
    /// 128 keeps >97% of the achievable work reduction at a quarter of the
    /// launch overhead of 32. `0` disables blocking entirely (one call for the
    /// whole chunk — the pre-2026-07 behavior), which is the kill switch if
    /// this is ever implicated in a numerics or latency regression.
    static let queryBlockSize: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_ATTN_QUERY_BLOCK"],
            let value = Int(raw), value >= 0
        else { return 128 }
        return value
    }()

    /// Query-block width used when a prompt pass is already on the blocked
    /// path and the layer's heads are wide (head dim 256 or 512). Wide heads
    /// do not enter the fused SDPA path, so a blocked prompt pass materializes
    /// one score rectangle per block; a narrower block holds a smaller
    /// rectangle. Only the grouping of the query rows changes: every row's
    /// softmax reduction still runs over the whole key axis, in the same
    /// order, so the produced values are unchanged.
    ///
    /// `0` disables the specialization (the block width falls back to
    /// `queryBlockSize`), which is the kill switch.
    static let wideHeadQueryBlockSize: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_ATTN_QUERY_BLOCK_WIDE"],
            let value = Int(raw), value >= 0
        else { return 64 }
        return value
    }()

    /// Block width for one attention call: the wide-head width when the call
    /// is on the blocked-query prompt path (`L > queryBlockSize`) and the
    /// layer's head dim is 256 or 512; the configured width otherwise. Decode
    /// has `L == 1` and can never enter. An explicit
    /// `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK` override that moves the configured
    /// width off its default also returns the configured width.
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

    /// Ceiling, in MiB, on the K+V a PACKED prefill may restack on the batch
    /// axis for one batched SDPA (`batchedPackedAttention`).
    ///
    /// Batching the rows is a strict dispatch win — 8 rows x 8 query blocks
    /// becomes 8 dispatches — but `concatenated` materializes a second copy
    /// of the rows' committed K/V. That copy is bounded by this budget so a
    /// long-context packed continuation falls back to the per-row
    /// decomposition instead of doubling a multi-GiB KV footprint the
    /// provider's `UnifiedMemoryCap` never reserved for it. A fresh 8 x 1024
    /// prompt pass is ~32 MiB and always batches.
    ///
    /// `0` disables batching entirely — the kill switch back to the pinned
    /// per-row path if this is ever implicated in a regression.
    static let packedBatchKVBudgetBytes: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_ATTN_BATCH_KV_BUDGET_MB"],
            let value = Int(raw), value >= 0
        else { return 512 << 20 }
        return value << 20
    }()

    /// PREFILL-PACKED-KV-ALIAS. R2 retains the parity-proven R1 executable path.
    /// When every row of a packed prefill was FRESH
    /// (no committed history), the per-row views `update` hands back are
    /// byte-for-byte the incoming batched K/V rectangles, and the
    /// `concatenated` restack in `batchedPackedAttention` re-materializes a
    /// copy of tensors the caller already holds batched. Attend the original
    /// rectangles instead. `0`/`false`/`no`/`off` restores the established
    /// restack, which also stays the path for every input the alias refuses.
    static let packedKVAliasEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_PACKED_KV_ALIAS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Join prompt attention blocks directly into the token-major layout that
    /// the following output projection consumes. The returned value remains a
    /// head-major view, so callers keep the same typed interface while their
    /// existing transpose restores the contiguous token-major buffer.
    static let tokenMajorJoinEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_TOKENMAJOR_JOIN"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// ATT-008 opt-in switch: batch-wide FULL-attention decode over pooled
    /// KV (`DARKBLOOM_GEMMA4_BATCHED_FULL_ATTENTION=1` enables it).
    /// DEFAULT OFF: three counterbalanced local B=8 probe pairs measured the
    /// consolidation at +0.27 ms/round (+1.2%) — the concurrent Metal
    /// encoder already overlaps the per-row dispatches it removes. Kept
    /// selectable because the mechanism is parity-proven bit-exact and the
    /// balance could differ on other hardware.
    static let batchedFullDecodeEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_BATCHED_FULL_ATTENTION"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// Whether a chunk of `L` queries should be split into blocks. Single
    /// queries (decode) and chunks already at or below the block width take
    /// the unchanged single-call path, so decode is provably untouched.
    @inline(__always)
    static func shouldBlockQueries(_ L: Int) -> Bool {
        queryBlockSize > 0 && L > queryBlockSize
    }

    /// The sink logits an `MLXFast.scaledDotProductAttention` call may be
    /// handed for `queryDType` activations.
    ///
    /// MLX requires the sink dtype to PROMOTE to the SDPA output dtype. fp16
    /// queries with fp32 sinks are therefore not a rounding wart but a
    /// process abort: MLX throws `[scaled_dot_product_attention] Type of
    /// sinks must promote to output type float16` in C++, mlx-c routes it to
    /// the installed error handler, and mlx-swift's handler calls
    /// `fatalError` (`MLX/ErrorHandler.swift`) — SIGTRAP, not a Swift error
    /// anything upstream can catch. A daemon hosting several models loses
    /// every in-flight request and emits nothing.
    ///
    /// Every SDPA terminal on a SERVING path funnels through here: the two
    /// in this file via `dispatchSinks`, and the paged prefill terminal
    /// (`PagedLayerCache.attendQueryBlock`) via that file's `prefillSinks`.
    /// Both callers hoist the cast to their top-level dispatch, so the
    /// per-row / per-block / per-token loops below them re-use one array.
    ///
    /// The remaining SDPA calls in the module are in `PagedDecodeProfiler`
    /// and `PagedBackendBenchmark`, which mint their own fixtures in the
    /// dtype they then query with; they are measurement rigs, not a path any
    /// request reaches.
    ///
    /// NOT for `PagedAttentionReference.composedAttention`: it runs in fp32
    /// throughout and widens the sinks itself, so narrowing them first would
    /// be a real precision loss. Both callers carve that case out on
    /// `softcap != nil`, which is what selects the composed path.
    @inline(__always)
    static func sdpaSinks(_ sinks: MLXArray?, queryDType: DType) -> MLXArray? {
        // `asType` returns `self` when the dtypes already match, so the
        // models shipping today (gpt-oss loads `sinks` in checkpoint dtype,
        // matching the activations) add no operation at all.
        sinks?.asType(queryDType)
    }

    /// The sinks every terminal reached by ONE top-level dispatch receives.
    ///
    /// Computed once per `updateAndAttend` / `attendBorrowing` /
    /// `updateAndAttendLastQuery` call rather than at the terminal: eager
    /// decode enters `attend` once per row, query-blocked prefill once per
    /// block and the MTP serial path once per query, so casting at the
    /// terminal rebuilds the same one-element conversion for every row,
    /// block, layer and generated token on latency-sensitive paths.
    ///
    /// Slicing preserves dtype, so the top-level `queries.dtype` is exactly
    /// the dtype every per-row / per-block slice below presents.
    @inline(__always)
    private static func dispatchSinks(
        _ sinks: MLXArray?, kind: CBv2LayerKind, queries: MLXArray, softcap: Float?
    ) -> MLXArray? {
        guard kind.hasSinks, let sinks else { return nil }
        // A softcap sends BOTH phases to the composed fp32 reference, which
        // wants the model's own (possibly wider) sinks — see `sdpaSinks`.
        return softcap == nil ? sdpaSinks(sinks, queryDType: queries.dtype) : sinks
    }

    /// Gathered-key columns ONE query block may see: from the EARLIEST
    /// query's window floor to the LATEST query's own position, so the
    /// block's queries are exactly the trailing `count` entries of the span
    /// — the invariant `maskMode` assumes.
    ///
    /// `historyCount` is the number of keys that predate the block's chunk;
    /// it means the same thing on both backends. Contiguous derives it as
    /// `keys.dim(2) - newTokenCount`, paged reads it off the assembled
    /// `PrefillKV`.
    ///
    /// **This arithmetic IS the activation-reserve bound.** Blocking is what
    /// pins the materialized score tensor at `[B, heads, count, kL]` instead
    /// of `[B, heads, L, kL]`, which is what lets `prefillChunkSize` grow
    /// without the flat 3 GiB reserve in the provider's `UnifiedMemoryCap`
    /// becoming a lie. On a sliding layer the span saturates at
    /// `window - 1 + blockSize` however long the chunk is; on a full layer
    /// only the query extent is capped. Both backends therefore have to
    /// compute the SAME span — if the two copies drift, one of them silently
    /// attends a different set of keys, and the memory bound stops holding
    /// on whichever side widened.
    ///
    /// It is shared rather than copied for exactly that reason. The paged
    /// track originally copied it because `attendQueryBlocks` is `private`
    /// and returns a symbolic mask mode — true, but a reason that applies to
    /// the attention CALL, not to the bounds.
    static func queryBlockBounds(
        historyCount: Int, offset: Int, count: Int, window: Int?
    ) -> (visibleStart: Int, visibleEnd: Int) {
        let visibleEnd = historyCount + offset + count
        let visibleStart = window.map { max(0, historyCount + offset + 1 - $0) } ?? 0
        return (visibleStart, visibleEnd)
    }

    /// Mask mode for a single-request attention call.
    ///
    /// - `L == 1` (decode): `.none`. The row's retained KV IS its window —
    ///   sliding-window eviction already dropped everything outside it.
    /// - `L > 1` (prefill chunk) against `kL` returned KV entries:
    ///   - `.causal` when no window is configured, or when `kL <= window`
    ///     (the window cannot bind: the oldest returned entry is inside
    ///     every query's window).
    ///   - causal ∧ window ARRAY mask when `kL > window` (a windowed layer's
    ///     multi-token update returned pre-eviction history so early chunk
    ///     tokens see their full window; later tokens must not over-attend).
    ///
    /// Pure in (L, kL, window): the same request produces the same mask mode
    /// at the same point in its lifetime regardless of batchmates.
    static func maskMode(L: Int, kL: Int, window: Int?, bidirectional: Bool = false)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        if L == 1 { return .none }
        if bidirectional {
            return .array(boolMask(L: L, kL: kL, window: window, bidirectional: true)!)
        }
        if let window, kL > window {
            // Relative coordinates: keys span [0, kL), queries are the last
            // L positions. Window comparisons are translation-invariant, so
            // the absolute offset is irrelevant.
            return .array(createCausalMask(n: L, offset: kL - L, windowSize: window))
        }
        return .causal
    }

    /// Update each row with this step's K/V and attend.
    ///
    /// - queries/keys/values: `[B, heads, L, headDim]` with `L == 1` for
    ///   decode, `B == 1` for a singleton prefill chunk, or rectangular
    ///   `[B > 1, L > 1]` packed prefill / MTP verification. Every row
    ///   dispatches against its own KV and optional span context.
    /// - softcap: construction-time attention-logit soft cap
    ///   (`cap * tanh(qk / cap)` before softmax, Gemma-2 style). When set,
    ///   BOTH phases take the composed reference path (SDPA cannot express
    ///   softcapping) — still one pinned path per phase.
    /// - spanContexts: when bound, one optional context per row. Non-nil
    ///   entries select that row's vision overlay; nil entries retain q=128
    ///   query-block causal/window semantics in the same rectangular call.
    /// - Returns `[B, queryHeads, L, headDim]`.
    static func updateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float? = nil,
        spanContexts: [CBv2SpanChunkContext?]? = nil,
        serializeQueries: Bool = false,
        decodeRingWriteFence: CBv2DecodeRingWriteFence? = nil,
        allowFusedRingWrite: Bool = false
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
                    // WRITE-016: fold this step's one-token ring write into
                    // ring pass A. The separate `decodeRingWrite` below is a
                    // `SliceUpdate` over a 4 MiB allocation the direct-ring
                    // attention graph still retains, so it cannot generally
                    // donate: 4 KiB of new K/V costs a full-ring copy, 25
                    // sliding layers x 8 rows x K/V per forward. The fused
                    // pass A stores the same bytes into the same evicted slot
                    // in place and serves logical token 1023 from the new K/V
                    // arrays, so no block reads the slot it writes and the
                    // accumulation order is unchanged. Refused (and skipped
                    // entirely, write included) unless the storage-owning
                    // layer has no K/V borrower that must keep observing the
                    // pre-write allocation.
                    if fusedRingWriteEnabled, allowFusedRingWrite,
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
                    if views.count == B,
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

                // Before every row's sliding ring reaches 1024 entries, retain
                // the established row-local SDPA path over the views just
                // returned by the updates above.
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

            // WRITE-016-D512: the D512 chain with the new token's K/V stored
            // in place by the QK dispatch (fence-chained like WRITE-016)
            // instead of 16 copy-on-write slice appends. Fails closed to the
            // append-then-attend call below (kill switch:
            // DARKBLOOM_GEMMA4_D512_FUSED_WRITE=0).
            // WRITE-022: the append as its own fenced store dispatch ahead of
            // the byte-for-byte stock D512 chain (samfenwick's db4ef5e design,
            // re-implemented with credit) — removes the same copies as the v2
            // fold below without its inner-loop addressing cost.
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

            // D512-SDPA: batched 3-dispatch full-attention decode with the
            // unfused chain's exact numerics (kill switch:
            // DARKBLOOM_GEMMA4_D512_DECODE_SDPA=0). Precedes ATT-008 so rows
            // stay unpooled; pooled rows fail its gate closed.
            if let output = CBv2RaggedComposedD512DecodeAttentionV1.updateAndAttend(
                rows: rows, kind: kind,
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: effectiveSinks, softcap: softcap)
            {
                CBv2EngageMark.once("d512sdpa")
                return output
            }

            // ATT-008: batch-wide FULL-attention decode. One pooled append +
            // one batched call replaces 8 per-row appends + 8 row-local
            // composed SDPA graphs, with bit-identical per-row numerics (see
            // `batchedFullDecodeUpdateAndAttend`). Fails closed to the
            // established per-row loop below, which stays correct on pooled
            // and unpooled rows alike.
            if let output = batchedFullDecodeUpdateAndAttend(
                rows: rows, kind: kind,
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: effectiveSinks, softcap: softcap)
            {
                CBv2EngageMark.once("att008")
                return output
            }

            // Batched decode: split queries per row, per-row update + SDPA
            // against that row's own KV, then concatenate. No masks — each row
            // sees exactly its own KV, so batch-composition invariance holds by
            // construction and fully-masked rows cannot exist.
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

        // Rectangular [B > 1, L > 1] packed prefill or MTP verify: every row
        // takes the same per-row path as a singleton chunk and attends
        // exactly its own KV. Serialized queries remain MTP-only; ordinary
        // packed prefill keeps q-blocking and optional row-local span masks.
        if serializeQueries {
            return packedPerRow(batch: B) { index, slice in
                updateAndAttendRowSerialQueries(
                    row: rows[index], kind: kind,
                    queries: slice(queries), keys: slice(keys), values: slice(values),
                    scale: scale, sinks: effectiveSinks, softcap: softcap)
            }
        }

        // Commit every row's K/V FIRST, in row order. `update` appends to that
        // row's own ring and hands back that row's own view; it reads nothing
        // from any other row and nothing from the attention that used to sit
        // between two commits. Hoisting the commits above the attention loop
        // therefore leaves each row seeing exactly the view it saw before, and
        // it is what lets the uniform case below see all B views at once.
        var cachedKeys: [MLXArray] = []
        var cachedValues: [MLXArray] = []
        cachedKeys.reserveCapacity(B)
        cachedValues.reserveCapacity(B)
        for (index, row) in rows.enumerated() {
            let (rowKeys, rowValues) = row.update(
                keys: keys[index ..< (index + 1)],
                values: values[index ..< (index + 1)])
            cachedKeys.append(rowKeys)
            cachedValues.append(rowValues)
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

    /// One batched SDPA for a packed `[B > 1, L > 1]` pass whose rows all hold
    /// the SAME committed K/V length and carry no span overlay.
    ///
    /// Why this is the same computation, not an approximation of it:
    /// `maskMode` is a pure function of `(L, kL, window)` and
    /// `attendQueryBlocks` derives each block's key span from
    /// `(historyCount, offset, count, window)` alone — `historyCount` being
    /// `kL - L`. Equal `kL` across rows therefore means every row would be
    /// handed the IDENTICAL mask and slice the IDENTICAL key columns, so the
    /// only thing separating B one-row calls from one B-row call is the batch
    /// extent. SDPA reduces independently per `(batch, head, query)`, so no
    /// row's sum acquires a term from a batchmate and none of them is
    /// re-ordered. What it buys is dispatches: an 8-row, 8-block prompt layer
    /// goes from 64 SDPA launches to 8.
    ///
    /// Returns nil — and the caller keeps the per-row decomposition — when the
    /// rows are NOT interchangeable this way: unequal committed lengths (mixed
    /// prefix reuse or continuation offsets), a mismatched head count/head dim
    /// /dtype, any bound span context, or a restack that would exceed
    /// ``packedBatchKVBudgetBytes``.
    ///
    /// PREFILL-PACKED-KV-ALIAS. `newKeys`/`newValues` are the caller's
    /// ALREADY-BATCHED `[B, kvHeads, n, headDim]` rectangles for the chunk the
    /// per-row `update` calls just committed (nil on the borrow path, which
    /// commits nothing). When every row's returned view spans exactly `n`
    /// tokens, every row was fresh, and the restack below rebuilds a
    /// byte-identical copy of those rectangles:
    ///   * both contiguous backends return `history ++ chunk` — the windowed
    ///     ring returns the chunk tensor ITSELF when `historyCount == 0`
    ///     (`kParts == [newKeys]`), and the full buffer returns
    ///     `storage[..., ..<offset, :]`, which after a fresh append holds
    ///     exactly the chunk's bytes;
    ///   * `kL == n` forces `historyCount == 0` on every row (returned length
    ///     is always `history + n`), so `concatenated(cachedKeys, axis: 0)`
    ///     reassembles the row slices of `newKeys` in row order — `newKeys`.
    /// Attending the original rectangles therefore feeds the SAME bytes to
    /// the same kernels and deletes the restack's full K+V round trip — at
    /// the ranked 8 x 1024 seed geometry, ~134 MB of copy traffic per sliding
    /// layer on the critical path (attention cannot start before the copy).
    /// `contiguous()` pins the layout contract: a row-contiguous rectangle
    /// (every fused QKV-norm kernel output) shares its buffer at zero cost,
    /// and anything else (a transposed-view fallback producer) materializes
    /// ONE copy — the exact cost of the restack this replaces — so no
    /// downstream q-block GEMM ever sees a layout the restacked path would
    /// not have produced. Kill switch:
    /// `DARKBLOOM_CBV2_PREFILL_PACKED_KV_ALIAS=0`.
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

        // PREFILL-PACKED-KV-ALIAS: every-row-fresh admission (see the doc
        // comment above). Anything short of the exact identity — a committed
        // history on any row (`kL > n`), a batch/head/dim/dtype mismatch, or
        // no rectangles at all (borrow path) — falls through to the
        // established restack.
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

    /// Attention for K/V that is ALREADY committed — verbatim the tail of
    /// `updateAndAttendRow`, so a batched call and a per-row call select the
    /// same path from the same `(L, kL, window)` and differ only in how many
    /// rows ride the batch axis.
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

    /// The batch-axis decomposition of a PACKED `[B, ...]` prompt pass —
    /// the one definition both backends share, so "packed" cannot come to
    /// mean two different things depending on storage.
    ///
    /// Row `index` sees the `index`-th batch slice of every tensor and
    /// nothing else; the per-row results are rejoined on axis 0 in row
    /// order. What is deliberately NOT shared is the body: the contiguous
    /// per-row path updates the row and masks in RELATIVE coordinates
    /// (`maskMode`, which may return symbolic `.causal`), while the paged
    /// one gathers pages before writing and masks in ABSOLUTE ones (always
    /// `.array` — MLX #3384). Those are genuinely different computations
    /// over genuinely different storage; forcing one body on both would
    /// break the paged file's pinned mask contract. The decomposition is
    /// the part that must not drift, so the decomposition is what is
    /// factored out.
    ///
    /// `batch == 1` passes the caller's tensors through UNSLICED and skips
    /// the concatenate, so a singleton call is provably the same graph it
    /// was before packing existed.
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

    /// Commit a full-attention prefill chunk's K/V while evaluating only its
    /// newest query (see LastQueryPrefillV2.swift). The newest causal query
    /// sees every key the chunk just wrote, so this is exactly the final row
    /// of ordinary chunk attention — mask-free by construction, which is why
    /// a bound span overlay cannot change it either.
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

    /// One row's update + attention — verbatim the single-request ([1, L])
    /// logic, shared by the B == 1 path and the rectangular [B > 1, L > 1]
    /// verify loop so a batched row is bit-identical to running alone.
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

    /// Attend against `sourceRows`' KV WITHOUT updating (Gemma-4 cross-layer
    /// KV sharing: shared layers project Q only and borrow the source
    /// layer's K/V — the source layer already appended this step's tokens
    /// earlier in the forward pass).
    ///
    /// View selection is a pure function of L (pinned-path discipline):
    ///  - CHUNK borrow (`L > 1`): `chunkBorrowViews(of:)`, NOT `snapshot()`.
    ///    After a windowed source's multi-token update, `snapshot()` is the
    ///    POST-eviction ring (≤ window entries), while the source layer's
    ///    own attention saw the PRE-eviction history + chunk
    ///    (`window - 1 + n` entries) so the chunk's earliest queries keep
    ///    their full window. Borrowing must attend the same view, with the
    ///    same `kL > window` array-mask path (the paged backend is already
    ///    exact here by construction — this mirrors its semantics).
    ///  - DECODE borrow (`L == 1`): `snapshot()`. The retained ring IS the
    ///    window (the source's same-step decode update already evicted),
    ///    so the mask-free decode path stays exact.
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
            // Rectangular [B > 1, L > 1] packed prefill / MTP verify:
            // independently borrow each source row's current chunk views.
            // Borrowing mutates nothing, so when those views are uniform the
            // same batch-axis stacking the storage-owning path uses applies
            // here unchanged (`batchedPackedAttention` explains why that is
            // the same computation). Gemma 4's KV-sharing layers are the
            // majority of the tower, so this is where most of the dispatch
            // collapse lands.
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
                // Ordinary decode borrows the retained ring. A staged serial
                // MTP transaction has not written that ring yet, so borrow
                // the source layer's logical post-update view instead.
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

    /// One row's chunk borrow + attention — verbatim the single-request
    /// ([1, chunk]) borrow logic, shared by the B == 1 path and the
    /// rectangular [B > 1, L > 1] verify loop.
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
            // Same overlay as the storage-owning path: the MLXVLM reference
            // applies it to the masks KV-shared layers reuse.
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

    /// The views a borrowing layer must attend for the current PREFILL-CHUNK
    /// step: the windowed source's step-scoped pre-eviction views (see
    /// `CBv2WindowedSequenceKV.borrowableViews()`), else `snapshot()`
    /// (full-attention sources retain everything).
    private static func chunkBorrowViews(of row: CBv2SequenceKV) -> (MLXArray, MLXArray) {
        if let windowed = row as? CBv2WindowedSequenceKV {
            return windowed.borrowableViews()
        }
        let (keys, values, _) = row.snapshot()
        return (keys, values)
    }

    // MARK: - Private

    /// ATT-008: one batched update + attend for the exact Gemma 4
    /// full-attention decode cohort (B=8, 16 query heads, 2 KV heads, D=512,
    /// bf16, scale 1.0, no sinks/softcap, mask-free L=1).
    ///
    /// D=512 has NO fused SDPA kernel (`sdpa_vector` supports 64/96/128/256),
    /// so `MLXFast.scaledDotProductAttention` always lowers these calls to
    /// the fast.cpp fallback graph: scale-multiply → GQA unflatten →
    /// `matmul` QKᵀ (gemv) → precise softmax → `matmul` scores·V (gemv_t).
    /// That graph is shape-generic in the batch extent, and for these M=1
    /// shapes every Metal dispatch decision it reaches — the gemv/gemv_t
    /// block configuration, the `gemv_al` alignment gate (requires
    /// `batch_size_out == 1`, never true here), the softmax variant and
    /// threadgroup size (functions of the key length only), and the
    /// `check_transpose` no-copy branches — depends only on
    /// (M, N, K, dtype, last-two-dim strides), never on the batch extent,
    /// which only scales `grid.z` / the row count. Issuing ONE B=8 call over
    /// the pooled `[8, 2, kL, 512]` views therefore reproduces each row's
    /// per-output add order bit-exactly BY CONSTRUCTION (verified uint16-
    /// identical against the per-row chain at kL ∈ {1024, 1025, 1100, 1152}
    /// and across simulated append steps).
    ///
    /// Per full layer per decode step this replaces 8×2 per-row cache slice
    /// assignments + 8 row-local 4-dispatch attention graphs + 1 output
    /// concat (~49 dispatches) with 2 slice assignments + 1 batched 4-dispatch
    /// graph (~6), with zero per-step copies (the pool is written in place;
    /// rows migrate once per cohort).
    ///
    /// Fails closed (returns nil, caller keeps the pinned per-row loop) on
    /// any other batch size, geometry, dtype, scale, sinks, softcap,
    /// bidirectional kind, non-`CBv2FullSequenceKV` row, unpoolable rows, or
    /// rows whose offsets are not in lockstep (e.g. after a speculative
    /// rollback of a subset). The per-row loop remains bit-identical on
    /// pooled storage, so falling back mid-stream is always safe.
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

        // Lockstep gate: one pooled append writes ONE slot for all rows, so
        // every row must be at the same committed offset with headroom.
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

    /// The only shape for which the custom batch-wide dispatch is a literal
    /// transcription of the row-local MLX two-pass kernels. All other models,
    /// phases, masks, sequence lengths, and dtypes fail closed in the custom
    /// helper and retain the established dispatch below.
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

    /// Attend a prompt chunk in QUERY BLOCKS, slicing K/V to each block's own
    /// visible span instead of computing the whole `[L, kL]` rectangle and
    /// masking the excess away.
    ///
    /// Two things this buys, both of which matter:
    ///
    /// 1. WORK. A sliding-window layer's chunk attention today evaluates
    ///    `L x (window - 1 + L)` score positions to use `L x window` of them:
    ///    a `(window - 1 + L)/window` ratio, i.e. 1.499x at L=512/window=1024
    ///    and 2.67x at L=2048. Per block the ratio collapses to
    ///    `(window - 1 + q)/window` — 1.062x at q=128 — because a block only
    ///    ever reads the keys its own queries can see.
    /// 2. MEMORY. The composed attention path (which Gemma 4 always takes:
    ///    MLX's fused kernel supports head_dim 64/80/128 and Gemma 4 uses
    ///    256/512) MATERIALIZES the `[B, heads, L, kL]` score tensor. Blocking
    ///    pins that at `[B, heads, q, kL]`, making it O(1) in chunk length —
    ///    which is what lets `prefillChunkSize` grow without the flat 3 GiB
    ///    activation reserve in `UnifiedMemoryCap` becoming a lie. At 124k
    ///    context a full-attention layer drops from 2.04 GB to 0.51 GB.
    ///
    /// Numerics: NOT bit-identical to the single-call path. The same non-zero
    /// terms are summed (masked entries contribute `exp(-inf) = 0`), but a
    /// different `kL` changes the reduction tiling and may select a different
    /// kernel specialization, so results can differ in the last ulp.
    ///
    /// Scope: this applies to every multi-token prompt call that reaches
    /// `updateAndAttendRow` / `borrowAndAttendRow`, including rectangular
    /// packed prefill. Vision rows keep the same q=128 blocking; each query
    /// block expands its K/V slice only as needed to include complete image
    /// spans touched by that block, then composes causal/window and
    /// bidirectional-span masks. Decode (`L == 1`) remains outside this path.
    ///
    /// `blockSize == 1` reproduces the MTP serial-query path exactly.
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
        // PREFILL-QSCALE-ELIDE: split the query head axis ONCE on the
        // row-contiguous chunk (a pure view) so no q-block has to be
        // materialized for the composed path. nil keeps every block on the
        // established `MLXFast.scaledDotProductAttention` call.
        // `blockSize <= 8` is the pinned MTP serial-query path (and any other
        // narrow verify block): those calls never reach the composed arm, so
        // build no graph nodes for them at all.
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
                    // Bidirectional span attention can see future keys and,
                    // on windowed layers, older same-span keys. Retain the
                    // complete touched span while keeping all other keys at
                    // the ordinary q-block bounds.
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
        // PREFILL-TOKENMAJOR-JOIN. The established head-major join is
        // immediately transposed and reshaped by Gemma4Attention, forcing a
        // second full pass over the prompt attention output. Permuting each
        // block before concatenation writes the final token-major rectangle
        // directly. Returning its inverse-transposed view preserves this
        // function's `[B, H, L, D]` contract; the caller's existing transpose
        // composes to identity and its reshape can share the buffer.
        //
        // `blockSize > 8 && newTokenCount > 8` is the prompt-only composed
        // attention plane. Decode is L=1 and MTP serial verification uses
        // blockSize=1, so neither can enter this branch.
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

    /// One query at a time — the pinned MTP serial-verification path.
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

    /// Single-request attention dispatch. Without a softcap this is MLXFast
    /// SDPA with `maskMode(L:kL:window:)`; with a softcap it is the composed
    /// fp32 reference (SDPA cannot express logit softcapping) with the
    /// EQUIVALENT boolean mask — both are pure functions of (L, kL, window),
    /// so each configuration keeps exactly one pinned path per phase.
    private static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        L: Int, kL: Int, window: Int?, sinks: MLXArray?, softcap: Float?,
        bidirectional: Bool = false, queryPlaneSlice: MLXArray? = nil
    ) -> MLXArray {
        // A model may widen Q for safer attention math while retaining compact
        // K/V storage. SDPA requires one dtype, so widen only these views.
        let attentionKeys = keys.dtype == queries.dtype ? keys : keys.asType(queries.dtype)
        let attentionValues =
            values.dtype == queries.dtype ? values : values.asType(queries.dtype)
        guard let softcap else {
            // Sinks arrive ALREADY normalized to the query dtype: see
            // `sdpaSinks` (MLX aborts the process on a wider sink) and
            // `dispatchSinks` (the conversion is hoisted to the top-level
            // call, so this terminal never rebuilds it per row or block).
            assert(
                sinks == nil || sinks!.dtype == queries.dtype,
                "CBv2AttentionV1: sinks must be normalized to the query dtype before SDPA")
            // PREFILL-QSCALE-ELIDE: on the prompt plane MLX takes its unfused
            // fallback (head dim 256/512, L > 8) whose first op is an identity
            // `scale * q` at Gemma 4's `scale == 1.0`. The transcription below
            // is that fallback with the identity deleted; it refuses every
            // input it cannot prove is the same graph.
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

    /// Boolean causal(∧window) mask (true == attend) equivalent to
    /// `maskMode(L:kL:window:)`, for the composed softcap path. nil for
    /// decode (L == 1: the retained KV IS the window).
    static func boolMask(
        L: Int, kL: Int, window: Int?, bidirectional: Bool = false
    ) -> MLXArray? {
        guard L > 1 else { return nil }
        // Relative coordinates: keys span [0, kL), queries are the last L.
        let qpos = MLXArray(Int32(kL - L) ..< Int32(kL)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(kL)).expandedDimensions(axis: 0)
        var mask = kpos .<= qpos
        if let window, kL > window {
            mask = mask .&& (kpos .> (qpos - Int32(window)))
        }
        if bidirectional {
            // Complete only the trailing current-chunk square. Cached prefix
            // columns keep their established causal/window visibility.
            var reverse = (kpos .>= qpos) .&& (kpos .>= Int32(kL - L))
            if let window {
                reverse = reverse .&& (kpos .< (qpos + Int32(window)))
            }
            mask = mask .|| reverse
        }
        return mask
    }

    // MARK: - Vision span-chunk path (pinned; span-containing chunks only)

    /// Zero-copy selection of the coalesced image blocks that can affect one
    /// query slice. Context blocks are sorted/non-overlapping by resolution,
    /// so intersections form one contiguous `ArraySlice`.
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

    /// Boolean causal(∧window) mask over arbitrary absolute query/key slices,
    /// overlaid only with the image blocks intersecting this query slice.
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

    /// Whole-chunk form retained for focused mask contracts.
    static func spanChunkMask(
        L: Int, kL: Int, window: Int?, context: CBv2SpanChunkContext
    ) -> MLXArray {
        precondition(L > 1, "span chunks are multi-token by construction")
        return spanMask(
            queryAbsoluteStart: context.chunkEnd - L, queryCount: L,
            keyAbsoluteStart: context.chunkEnd - kL, keyCount: kL,
            window: window, blocks: context.blocks[...])
    }

    /// Span-chunk attention dispatch: always an explicit boolean array mask
    /// (the bidirectional overlay cannot ride the symbolic `.causal` mode).
    /// One pinned path per configuration — plain SDPA, or the composed fp32
    /// reference when a softcap is configured (same split as `attend`).
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
            // Same contract as `attend`: normalized upstream by
            // `dispatchSinks`, because MLX traps on fp16 queries + fp32 sinks.
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
