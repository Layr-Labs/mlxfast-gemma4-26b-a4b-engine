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
        serializeQueries: Bool = false
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
            sinks: effectiveSinks, softcap: softcap, spanContexts: spanContexts)
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
    private static func batchedPackedAttention(
        kind: CBv2LayerKind, queries: MLXArray,
        cachedKeys: [MLXArray], cachedValues: [MLXArray],
        window: Int?, scale: Float, sinks: MLXArray?, softcap: Float?,
        spanContexts: [CBv2SpanChunkContext?]?
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
        if shouldBlockQueries(L) && !kind.isBidirectional {
            return attendQueryBlocks(
                queries: queries, keys: cachedKeys, values: cachedValues,
                newTokenCount: L, window: window, scale: scale,
                sinks: sinks, softcap: softcap, blockSize: queryBlockSize,
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
        if shouldBlockQueries(L) && !sourceKind.isBidirectional {
            return attendQueryBlocks(
                queries: queries, keys: cachedKeys, values: cachedValues,
                newTokenCount: L, window: window(of: sourceKind), scale: scale,
                sinks: sinks, softcap: softcap, blockSize: queryBlockSize,
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
                outputs.append(
                    attend(
                        queries: querySlice, keys: keySlice, values: valueSlice,
                        scale: scale, L: count, kL: visibleEnd - visibleStart,
                        window: window, sinks: sinks, softcap: softcap))
            }
            offset += count
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
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
        bidirectional: Bool = false
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
