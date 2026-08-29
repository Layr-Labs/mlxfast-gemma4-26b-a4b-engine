// PagedLayerCache.swift
//
// Batch-facing per-layer cache for the paged backend (WS-C): the
// `CBv2AttendingLayerCache` that owns BOTH the KV update and the attention
// dispatch, so models and the scheduler never see the storage layout.
//
// Attention paths (numerically pinned — one path per phase, never switching
// mask representation across steps):
//   - decode (`L == 1`, any B, including B == 1): the paged Metal kernel.
//     Per-row window clamping is start-offset arithmetic on absolute
//     positions computed host-side from Swift Ints; masks do not exist.
//   - prefill chunk (`B == 1`, `L > 1`): per-request SDPA over gathered
//     pages with an explicit BOOL mask built from absolute positions
//     (always `.array`, never `.none`/`.causal`, so the path cannot drift
//     — see MLX #3384). Models with an attention-logit softcap use the
//     composed reference path instead (SDPA cannot express softcap).
//     The chunk is attended in QUERY BLOCKS (`CBv2AttentionV1.queryBlockSize`,
//     shared kill switch `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK`), which bounds the
//     materialized score tensor at `[1, queryHeads, block, visible]` instead
//     of `[1, queryHeads, chunk, retained]`. The mask representation is
//     unchanged by blocking: every block still gets an absolute-position
//     BOOL `.array`.
//   - PACKED prefill (`B > 1`, `L > 1`, WS-2.1): the SAME per-request chunk
//     path, run once per row and concatenated on the batch axis — the
//     storage-agnostic rectangular loop `CBv2AttentionV1.updateAndAttend`
//     already uses, ported to paged storage. Each row gathers its OWN pages
//     and masks in its OWN absolute coordinates, so a packed row is
//     BIT-IDENTICAL to that row run alone and cross-row contamination is
//     impossible by construction rather than by mask arithmetic. Requires
//     WS-0.2p's query blocking to already bound the per-row score tensor,
//     which it does.
//   - VISION span chunk (`B == 1`, `L > 1`, spans bound through
//     `CBv2SpanMaskBinding`, WS-2.2): the same mask, OR-ed with a
//     bidirectional-within-block overlay. Because this file's mask is
//     ALREADY materialized in absolute coordinates, the overlay is two
//     comparisons against the block bounds — no chunk-end coordinate
//     reconstruction, and no escape from a symbolic mode. Span chunks stay
//     UNBLOCKED: a block's visible span is causal, and a bidirectional query
//     may attend keys AFTER its own position.
//   - MTP rectangular verification (`CBv2MTPRectangularSerializing`, `L > 1`
//     with `B` rows): the decode kernel again, one query COLUMN at a time,
//     so each column is bit-identical to that column run as a standalone
//     `L == 1` decode. No new kernel and no multi-query attention.
//
// Rows with different retained lengths are handled internally: the caller
// never pads and never builds masks.

import Foundation
import MLX
import MLXFast

public final class PagedLayerCache: CBv2AttendingLayerCache {
    public let layerIndex: Int
    public let kind: CBv2LayerKind
    let pool: PagedKVPool
    /// Optional attention-logit soft cap. Not part of the contract's
    /// `updateAndAttend` signature, so it is layer-cache configuration
    /// (from model config) instead.
    public let attentionSoftcap: Float?

    private var pagedRows: [PagedSequenceKV] = []

    // Per-row absolute RoPE offsets `[B]` (int32, device array). REBUILT from
    // host integers only on membership changes (`setRows`); ADVANCED
    // on-device (`+ L`) inside `updateAndAttend`, mirroring `CBv2LayerCache`.
    // The step loop therefore never uploads a fresh host array per layer per
    // step, and never syncs — the model reads this each step and feeds it
    // into the forward graph, which the step's `asyncEval` collapses so the
    // lazy `+ L` chain cannot grow O(steps).
    private var cachedPositionOffsets: MLXArray = MLXArray([] as [Int32])
    /// Times `positionOffsets` was rebuilt from host integers (test hook);
    /// must only move on batch membership changes, never inside the step loop.
    private(set) var positionOffsetsHostRebuilds = 0

    // Device block-table cache: rebuilt only when a row's page table
    // changes (page allocated / rollback / composition change), not on
    // every step. Fingerprinted by the pool-issued row SERIAL (never
    // reused), not ObjectIdentifier — a heap address can be recycled after
    // a finished request's row deallocates, and a same-shape decode would
    // then attend the FINISHED request's page ids (cross-request read).
    private var cachedTables: MLXArray?
    private var cachedTablesFingerprint: [(serial: UInt64, tableVersion: Int)] = []
    /// Times the device tables were rebuilt (test/telemetry hook).
    private(set) var tablesRebuildCount = 0

    // Kernel params `{softcap, scale, 0…}` are constant across steps for a
    // layer; cache the device array keyed on the scale actually passed.
    private var cachedParams: MLXArray?
    private var cachedParamsScale: Float?

    // Sinks prepared for the kernel (fp32, >= 8 elements). Models pass the
    // same parameter array every step, so key the cache on object identity.
    private var cachedSinks: MLXArray?
    private var cachedSinksSource: ObjectIdentifier?

    /// WS-3.4 (`CBv2MTPRectangularSerializing`): while set, a rectangular
    /// `[B, *, L > 1, *]` call is attended one query column at a time, so
    /// each column is bit-identical to that column run as a standalone
    /// `L == 1` decode. The engine sets it for the duration of an MTP
    /// verification round and clears it in a `defer`.
    var mtpSerializesRectangularAttention = false

    /// WS-1.2. The KV a KV-shared sibling needs in order to attend THIS
    /// layer's most recent prompt chunk, ONE ENTRY PER ROW of that chunk
    /// pass (packed prefill runs `B` rows through it — WS-2.1). A borrower
    /// runs AFTER the source wrote, so it cannot reproduce the pre-write
    /// view by gathering; the source has to keep it.
    ///
    /// LOAD-BEARING UNDER THE RING THAT SHIPPED, and free today only by
    /// model accident. Do not quote a formula here; read
    /// `PagedKVPool.ringPageCount`. Its CACHE bound no longer carries a
    /// `maxPrefillChunk` term — it sizes to `maxWindowExposure(window) +
    /// maxSpeculativeSpan` — precisely BECAUSE this file gathers a chunk
    /// before writing it. Under that ring a borrower can no longer
    /// re-derive the view by gathering after the fact: the older part of
    /// the chunk's window has already been overwritten by the chunk's own
    /// tail, and no gather order recovers it. This array is where the
    /// source keeps it.
    ///
    /// It costs nothing today only because neither supported model has a
    /// borrower: gemma-4 and gpt-oss both run `num_kv_shared_layers: 0`, so
    /// `LayerCacheBankV2` clears `retainsChunkForBorrowers` on every layer
    /// (see this file's `CBv2KVSourceChunkRetaining` conformance). That is a
    /// MODEL FACT, not a property of this design — a checkpoint with a
    /// non-zero shared-KV tail switches the retention on and then depends on
    /// it for correctness. Do not delete it as unused.
    private var retainedPrefillKV: [PrefillKV] = []
    /// Defaults to `true` so a cache used outside a bank is correct; the
    /// bank turns it off for every layer no sibling borrows — which is all
    /// of them for both supported models, so neither pays a byte.
    private var retainsChunkForBorrowers = true

    /// WS-2.2 (`CBv2SpanMaskBinding`). The bidirectional image spans of the
    /// ONE vision prefill chunk currently being built, in ABSOLUTE token
    /// coordinates — the same coordinates this file's masks already live in,
    /// so the overlay needs no translation. The engine binds it immediately
    /// before that chunk's graph build and unbinds it immediately after, so
    /// it is ALWAYS nil for decode, text chunks, and other requests' chunks.
    private(set) var boundSpanContext: CBv2SpanChunkContext?

    public init(
        layerIndex: Int, kind: CBv2LayerKind, pool: PagedKVPool,
        attentionSoftcap: Float? = nil
    ) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.pool = pool
        self.attentionSoftcap = attentionSoftcap
    }

    // MARK: - Rows

    public var rows: [CBv2SequenceKV] { pagedRows }

    /// Set the current batch rows (row order == batch row order). O(B);
    /// join = append a row object, leave = drop it — no storage moves.
    /// (Contract `CBv2AttendingLayerCache.setRows` — the canonical binding.)
    public func setRows(_ rows: [CBv2SequenceKV]) {
        precondition(
            kind.sharesKVWithLayer == nil || rows.isEmpty,
            "KV-shared layers own no rows; attention borrows via attendBorrowing")
        pagedRows = rows.map { row in
            guard let paged = row as? PagedSequenceKV else {
                fatalError("[PagedLayerCache] rows must be PagedSequenceKV from the same backend")
            }
            precondition(
                paged.groupKey == PagedKVGroupKey(kind),
                "row group \(paged.groupKey) does not match layer kind")
            return paged
        }
        rebuildPositionOffsets()
        retainedPrefillKV = []
    }

    /// Per-row absolute RoPE offsets `[B]` (device int32). Read BEFORE
    /// `updateAndAttend` for the step — it holds the PRE-update offsets of
    /// the tokens about to be processed (snapshot semantics). KV-shared
    /// layers own no rows, so their own value is empty (they reuse the
    /// source layer's pre-update capture).
    public var positionOffsets: MLXArray { cachedPositionOffsets }

    private func rebuildPositionOffsets() {
        positionOffsetsHostRebuilds += 1
        CBv2CoreInstrumentation.recordPositionOffsetsHostRebuild()
        cachedPositionOffsets = MLXArray(pagedRows.map { Int32($0.absoluteOffset) })
    }

    // MARK: - Attention

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer == nil, "shared layers must call attendBorrowing")
        let b = queries.dim(0)
        let l = queries.dim(2)
        precondition(b == pagedRows.count, "queries batch \(b) != rows \(pagedRows.count)")
        // Sinks apply only to layers that declare them (same gate as
        // CBv2AttentionV1) — a sink-free layer must ignore a passed array.
        let effectiveSinks = kind.hasSinks ? sinks : nil

        let output: MLXArray
        if l == 1 {
            // Decode: reserve each row's destination host-side; the kernel
            // writes the tiles IN PLACE before attending (fused write —
            // the slabs are never versioned through MLX ops).
            let targets = pagedRows.map { $0.prepareDecodeWrite() }
            retainedPrefillKV = []
            output = dispatchDecode(
                queries: queries,
                newKeys: keys.squeezed(axis: 2),
                newValues: values.squeezed(axis: 2),
                writeTargets: targets,
                rows: pagedRows, scale: scale, sinks: effectiveSinks)
        } else if mtpSerializesRectangularAttention {
            // MTP rectangular verification: the round batches the
            // weight-bound model body across the `1 + k` columns, but
            // attention is serialised per column. Each column reserves its
            // own destination and the kernel fuses the write, exactly as a
            // standalone decode step would. This branch is checked BEFORE
            // the prompt-chunk branch: an MTP round's `[B, 1 + k]` call is
            // rectangular VERIFICATION, never a packed prompt chunk.
            retainedPrefillKV = []
            output = attendRectangularColumns(
                queries: queries, keys: keys, values: values,
                rows: pagedRows, scale: scale, sinks: effectiveSinks)
        } else {
            precondition(
                boundSpanContext == nil || b == 1,
                "[PagedLayerCache] span-bearing chunks are never packed")
            // Prompt chunk. `b == 1` is the ordinary per-request shape;
            // `b > 1` is WS-2.1 PACKED prefill, which is the SAME work run
            // once per row. Nothing here reads a batchmate: the gather
            // walks THIS row's page table, the write lands in THIS row's
            // pages, and the mask is built from THIS row's absolute
            // positions. That is why a packed row is bit-identical to the
            // row run alone, and why cross-row contamination is not a
            // property the mask has to establish.
            //
            // WS-1.2: assemble each chunk's KV BEFORE writing it, so this
            // path only ever asks the ring for `window - 1` tokens. The
            // chunk's own K/V need no gather at all — they are the
            // arguments. A row in a FROZEN REPLAY is the one exception, and
            // it is why the write lives inside `prefillKVWritingChunk`
            // instead of out here: that row's chunk half comes from the
            // pages too, and it has to be read AFTER the cursor moves.
            //
            // Gathering AFTER the write instead asks for
            // `window - 1 + chunk` (`retainedCount` with
            // `lastUpdateTokens == chunk`), which is exactly the term that
            // would force `ringPageCount` to carry `maxPrefillChunk` again
            // and put gemma-4's windowed layers back at the 97 pages a
            // 1,024-token window used to cost. With the gather hoisted the
            // ring only has to satisfy `ringPages * pageSize >= window - 1`,
            // which is what let it drop to `window + maxSpeculativeSpan`
            // (65 pages; derive it from `ringPageCount`). Under that sizing a
            // post-write gather is not merely wasteful, it aborts the
            // process on `gatherRange`'s eviction precondition for any
            // windowed chunk past `pageSize + 1` tokens.
            //
            // The batch-axis decomposition is `CBv2AttentionV1.packedPerRow`,
            // shared with the contiguous backend so "packed" cannot mean two
            // different things depending on storage. Only the per-row BODY
            // is paged's own, and it has to be: contiguous updates the row
            // and masks in relative coordinates, this gathers pages before
            // writing and masks in absolute ones.
            var views: [PrefillKV] = []
            views.reserveCapacity(b)
            let chunkSinks = prefillSinks(effectiveSinks, queryDType: queries.dtype)
            output = CBv2AttentionV1.packedPerRow(batch: b) { index, slice in
                let row = pagedRows[index]
                let kv = prefillKVWritingChunk(
                    row: row, chunkKeys: slice(keys), chunkValues: slice(values),
                    dtype: queries.dtype)
                views.append(kv)
                return prefillAttend(
                    queries: slice(queries), kv: kv, scale: scale, sinks: chunkSinks)
            }
            retainedPrefillKV = retainsChunkForBorrowers ? views : []
        }
        // Advance offsets ON-DEVICE (uniform L for every row in the call:
        // decode is [B,1], a prompt chunk is [1,chunk], and a packed group
        // is [B,chunk] with one common chunk length) — the rows just
        // advanced their absolute counters by exactly L, so the cached
        // device array tracks them without a per-step host rebuild.
        cachedPositionOffsets = cachedPositionOffsets + Int32(l)
        return output
    }

    public func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer != nil, "attendBorrowing requires a KV-shared layer")
        guard let src = source as? PagedLayerCache else {
            fatalError("[PagedLayerCache] can only borrow from another PagedLayerCache")
        }
        precondition(
            kind.attention == src.kind.attention,
            "KV-shared layer must share the source layer's attention type")
        let l = queries.dim(2)
        // Same sink gate as updateAndAttend, keyed on THIS layer's kind.
        let effectiveSinks = kind.hasSinks ? sinks : nil
        if l == 1 {
            return dispatchDecode(
                queries: queries, rows: src.pagedRows, scale: scale, sinks: effectiveSinks,
                tableProvider: src)
        } else if mtpSerializesRectangularAttention {
            // Same per-column serialisation, minus the writes: the source
            // layer owns the KV and already wrote all L columns. That is
            // precisely why the columns need an explicit `qPosFromEnd` —
            // the source rows' `absoluteOffset` has ALREADY advanced by L,
            // so `decodeAttendRange` on its own would resolve every column
            // to the last one.
            return attendRectangularColumns(
                queries: queries, keys: nil, values: nil,
                rows: src.pagedRows, scale: scale, sinks: effectiveSinks,
                tableProvider: src)
        } else {
            let b = queries.dim(0)
            precondition(
                b == src.pagedRows.count,
                "borrowed chunk batch \(b) != source rows \(src.pagedRows.count)")
            // The source assembled `gather(window history) ++ chunk` before
            // it wrote, and the post-write ring cannot reproduce the older
            // part of it (WS-1.2), so the borrower attends the SOURCE's
            // retained views rather than re-gathering — one per packed row,
            // in row order. `retainsChunkForBorrowers` is only ever cleared
            // for a layer no sibling borrows from, so reaching this with no
            // retained views means the bank wired the KV-sharing graph
            // wrong.
            guard src.retainedPrefillKV.count == b else {
                fatalError(
                    "[PagedLayerCache] layer \(layerIndex) borrows from layer "
                        + "\(src.layerIndex), which retained \(src.retainedPrefillKV.count) "
                        + "chunk views for \(b) rows — the source was told it has no "
                        + "borrowers, or it last ran a decode step")
            }
            // Same shared batch-axis decomposition as the owning path.
            let chunkSinks = prefillSinks(effectiveSinks, queryDType: queries.dtype)
            return CBv2AttentionV1.packedPerRow(batch: b) { index, slice in
                let kv = src.retainedPrefillKV[index]
                precondition(
                    kv.queryCount == l,
                    "borrowed chunk is \(kv.queryCount) tokens, queries are \(l)")
                return prefillAttend(
                    queries: slice(queries), kv: kv, scale: scale, sinks: chunkSinks)
            }
        }
    }

    // MARK: - Decode path (paged kernel)

    /// One fused kernel dispatch for the batch. `newKeys`/`newValues`
    /// (`[B, kvHeads, D]`) with per-row `writeTargets` make pass A write
    /// the step's tiles in place before attending; both are nil on the
    /// KV-borrowing path (the owning layer already wrote).
    ///
    /// `qPosFromEnd` is how many positions BEFORE each row's frontier this
    /// query sits: 0 (the default) is the row's last written position, i.e.
    /// ordinary decode. Only MTP rectangular verification on a KV-borrowing
    /// layer passes a non-zero value.
    private func dispatchDecode(
        queries: MLXArray,
        newKeys: MLXArray? = nil,
        newValues: MLXArray? = nil,
        writeTargets: [(page: Int32, slot: Int)]? = nil,
        rows: [PagedSequenceKV], scale: Float, sinks: MLXArray?,
        tableProvider: PagedLayerCache? = nil,
        qPosFromEnd: Int = 0
    ) -> MLXArray {
        precondition((newKeys == nil) == (writeTargets == nil))
        let provider = tableProvider ?? self
        let group = pool.group(rows[0].groupKey)
        let tables = provider.deviceTables(rows: rows)

        let (seqinfo, maxAttendLength) = PagedAttentionKernel.seqinfo(
            rows.enumerated().map { i, row in
                row.seqInfoRow(
                    attending: attendRange(row: row, qPosFromEnd: qPosFromEnd),
                    writeTarget: writeTargets?[i])
            })

        let (out, nextFence) = PagedAttentionKernel.decode(
            queries: queries,
            newKeys: newKeys,
            newValues: newValues,
            kSlab: group.kSlab,
            vSlab: group.vSlab,
            tables: tables,
            seqinfo: seqinfo,
            maxAttendLength: maxAttendLength,
            sinks: preparedSinks(sinks),
            params: params(scale: scale),
            softcap: attentionSoftcap != nil,
            pageSize: pool.config.pageSize,
            writeFence: group.writeFence,
            kernelSource: pool.kernelSource
        )
        // The fused write advanced the group's write-fence chain: later
        // slab readers (KV-borrowing layers, next steps' dispatches,
        // gathers) consume it and order after this dispatch's writes.
        if let nextFence {
            group.writeFence = nextFence
        }
        // [B, QH, D] -> [B, QH, 1, D], back in the model dtype.
        var result = out.expandedDimensions(axis: 2)
        if result.dtype != queries.dtype {
            result = result.asType(queries.dtype)
        }
        return result
    }

    /// The absolute range a query `qPosFromEnd` positions before `row`'s
    /// frontier may attend, in the `(start, length)` form the kernel wants.
    ///
    /// `qPosFromEnd == 0` DELEGATES to the row's own `decodeAttendRange`, so
    /// the ordinary decode path keeps exactly one definition of the range
    /// and this generalisation cannot drift from it.
    private func attendRange(
        row: PagedSequenceKV, qPosFromEnd: Int
    ) -> (start: Int, length: Int) {
        guard qPosFromEnd > 0 else { return row.decodeAttendRange }
        let qPos = row.absoluteOffset - 1 - qPosFromEnd
        precondition(
            qPos >= row.baseOffset,
            "[PagedLayerCache] rectangular column \(qPosFromEnd) back from "
                + "\(row.absoluteOffset) precedes the row's base \(row.baseOffset)")
        var start = row.baseOffset
        if let window = row.windowSize {
            start = max(start, qPos - window + 1)
        }
        return (start, qPos - start + 1)
    }

    /// MTP rectangular verification (WS-3.4): attend `[B, *, L, *]` one
    /// query column at a time over the EXISTING fused decode dispatch.
    ///
    /// Non-nil `keys`/`values` mean this layer OWNS the KV: each column
    /// reserves its own destination (`prepareDecodeWrite`) and the kernel
    /// fuses the write, so column `t` attends `[max(base, qPos - window + 1),
    /// qPos]` with `qPos` its own absolute position — bit-identical to that
    /// column run as a standalone `L == 1` decode. Nil means a KV-borrowing
    /// layer whose source already wrote every column, so the columns walk
    /// BACKWARD from the source rows' already-advanced frontier.
    private func attendRectangularColumns(
        queries: MLXArray, keys: MLXArray?, values: MLXArray?,
        rows: [PagedSequenceKV], scale: Float, sinks: MLXArray?,
        tableProvider: PagedLayerCache? = nil
    ) -> MLXArray {
        precondition((keys == nil) == (values == nil))
        precondition(
            rows.count == queries.dim(0),
            "rectangular verification needs one row per query batch entry")
        let l = queries.dim(2)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(l)
        for t in 0 ..< l {
            let column = queries[0..., 0..., t ..< (t + 1), 0...]
            if let newKeys = keys, let newValues = values {
                let targets = rows.map { $0.prepareDecodeWrite() }
                outputs.append(
                    dispatchDecode(
                        queries: column,
                        newKeys: newKeys[0..., 0..., t ..< (t + 1), 0...].squeezed(axis: 2),
                        newValues: newValues[0..., 0..., t ..< (t + 1), 0...].squeezed(axis: 2),
                        writeTargets: targets,
                        rows: rows, scale: scale, sinks: sinks,
                        tableProvider: tableProvider))
            } else {
                outputs.append(
                    dispatchDecode(
                        queries: column, rows: rows, scale: scale, sinks: sinks,
                        tableProvider: tableProvider, qPosFromEnd: l - 1 - t))
            }
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }

    /// Kernel params `{softcap, scale, 0…}`, cached across steps.
    private func params(scale: Float) -> MLXArray {
        if let cached = cachedParams, cachedParamsScale == scale {
            return cached
        }
        var values: [Float] = [attentionSoftcap ?? 1.0, scale]
        values.append(contentsOf: [Float](repeating: 0, count: 8 - values.count))
        let params = MLXArray(values)
        cachedParams = params
        cachedParamsScale = scale
        return params
    }

    /// Sinks prepared for the merge kernel: fp32, padded to >= 8 elements
    /// so the generated signature keeps the `device` address space.
    private func preparedSinks(_ sinks: MLXArray?) -> MLXArray? {
        guard let sinks else { return nil }
        if let cached = cachedSinks, cachedSinksSource == ObjectIdentifier(sinks) {
            return cached
        }
        var s = sinks.asType(.float32).reshaped([-1])
        if s.dim(0) < 8 {
            s = concatenated([s, MLXArray.zeros([8 - s.dim(0)], dtype: .float32)])
        }
        cachedSinks = s
        cachedSinksSource = ObjectIdentifier(sinks)
        return s
    }

    /// Sinks prepared for the PREFILL SDPA terminal (`attendQueryBlock`),
    /// computed ONCE per `updateAndAttend` / `attendBorrowing` call.
    ///
    /// The narrowing itself is `CBv2AttentionV1.sdpaSinks` — the shared
    /// primitive that documents why fp16 queries with fp32 sinks are a
    /// SIGTRAP rather than a rounding wart. Only the hoisting is paged's
    /// own, and it has to be: `attendQueryBlock` runs once per query BLOCK
    /// per packed ROW, so casting at the terminal rebuilds the same
    /// one-element conversion for every row, block and layer of a chunk.
    /// Slicing preserves dtype, so the top-level `queries.dtype` is exactly
    /// the dtype every per-row / per-block slice presents.
    ///
    /// This is the SDPA terminal's contract only. The decode leg's
    /// `preparedSinks` (fp32, padded to 8) is the KERNEL's, and the two must
    /// not be merged: narrowing before `preparedSinks` widens back to fp32
    /// would round-trip fp32 sinks through fp16, and a fresh array per step
    /// would also miss that function's identity cache.
    private func prefillSinks(_ sinks: MLXArray?, queryDType: DType) -> MLXArray? {
        guard let sinks else { return nil }
        // A softcap routes `attendQueryBlock` to the composed fp32
        // reference, which widens the sinks itself — narrowing first would
        // be a real precision loss. Same carve-out as
        // `CBv2AttentionV1.dispatchSinks`.
        guard attentionSoftcap == nil else { return sinks }
        return CBv2AttentionV1.sdpaSinks(sinks, queryDType: queryDType)
    }

    /// Device `[B, maxPages]` int32 block tables, rebuilt only when some
    /// row's identity (pool serial) or page table changed since the last
    /// dispatch. Internal (not private) for regression tests.
    func deviceTables(rows: [PagedSequenceKV]) -> MLXArray {
        let fingerprint = rows.map { (serial: $0.serial, tableVersion: $0.tableVersion) }
        if let cached = cachedTables,
            fingerprint.count == cachedTablesFingerprint.count,
            zip(fingerprint, cachedTablesFingerprint).allSatisfy({ $0 == $1 })
        {
            return cached
        }
        tablesRebuildCount += 1
        // Pad to >= 8 columns so the kernel signature's address space is
        // stable across batch shapes (see PagedAttentionKernel), and pad
        // with the group's POISON page rather than the literal `0` (WS-0.5).
        // Page 0 is not a sentinel: the free list used to be built reversed
        // and popped with `removeLast()`, so page 0 was the FIRST page handed
        // to the first tenant, and a table slot padded with it named that
        // tenant's live KV. The poison page is reserved out of the free list,
        // refcount-pinned and permanently zeroed, so a slot the kernel should
        // never reach reads zeros instead of another request's tokens.
        let maxPages = max(8, rows.map { $0.table.count }.max() ?? 0)
        let poison = pool.poisonPage(group: PagedKVGroupKey(kind))
        var flat = [Int32](repeating: poison, count: rows.count * maxPages)
        for (i, row) in rows.enumerated() {
            flat.replaceSubrange(
                (i * maxPages) ..< (i * maxPages + row.table.count), with: row.table)
        }
        let tables = MLXArray(flat, [rows.count, maxPages])
        cachedTables = tables
        cachedTablesFingerprint = fingerprint
        return tables
    }

    // MARK: - Prefill path (per-request SDPA over `gather ++ chunk`)

    /// Exactly the KV one prompt chunk attends: the window history gathered
    /// from the ring BEFORE the chunk was written, followed by the chunk
    /// itself, plus the absolute position of the first key.
    ///
    /// Assembling this before the write is WS-1.2. It is also what a
    /// KV-borrowing sibling has to be handed, because after the write the
    /// older part of it is no longer in the ring.
    private struct PrefillKV {
        /// `[1, kvHeads, historyCount + queryCount, headDim]`.
        let keys: MLXArray
        let values: MLXArray
        /// Absolute position of key column 0.
        let start: Int
        /// Absolute position of the chunk's first query, which is also the
        /// first key column contributed by the chunk itself.
        let queryStart: Int

        /// Keys that predate the chunk — the index of the first query
        /// within `keys`, and the offset every block bound is shifted by.
        var historyCount: Int { queryStart - start }
        var queryCount: Int { keys.dim(2) - historyCount }
    }

    /// Write this chunk into `row` and return exactly the KV it attends.
    ///
    /// The write lives in here rather than at the call site because the two
    /// shapes below need the gather on OPPOSITE sides of it, and a caller
    /// that has to remember which order applies gets it wrong.
    ///
    /// ORDINARY (`frozenHighWater <= absoluteOffset` — every windowed row,
    /// and every row that never adopted a frozen prefix). Gather the window
    /// history BEFORE the write and splice on the chunk tensor the layer was
    /// handed. That order is WS-1.2: it is the only reason the ring can be
    /// `window + span` rather than `window - 1 + chunk + span`, and gathering
    /// after the write here aborts the process on `gatherRange`'s eviction
    /// precondition. The chunk's own K/V need no gather at all — they are
    /// the arguments.
    ///
    /// FROZEN REPLAY (`frozenHighWater > absoluteOffset`). Gather the WHOLE
    /// view, chunk included, from the row's pages AFTER the write. During
    /// prefix reuse's `.frozenFullReplay` the layer is handed projections
    /// computed by SLIDING rows that do not have their windows back yet, so
    /// the chunk tensor is POISONED — but this row's storage through M is the
    /// adopted, exact K/V, so the chunk's own positions can simply be read.
    /// That is what `CBv2FrozenReplayFullSequenceKV.update` does on the
    /// contiguous backend (it discards the replayed projections and returns
    /// the cached keys for the whole chunk, diagonal included), and matching
    /// it is what makes a frozen paged replay exact from the same position
    /// contiguous is — no `+ maxPrefillChunk` of extra replay.
    ///
    /// Four things make the post-write gather here safe, and it is safe ONLY
    /// because all four hold:
    ///  1. `PagedSequenceKV.adoptFrozen` refuses windowed rows, so a frozen
    ///     row has `ringPages == nil`. There is no ring, so no page the
    ///     chunk could have recycled and no eviction branch to trip.
    ///  2. `PagedSequenceKV.write` is CURSOR-ONLY below M: it advances
    ///     `absoluteOffset` and touches no storage. There is no
    ///     read-after-write to order, and the bytes read are the adopted
    ///     ones, written once at adoption and host-synced since.
    ///  3. Moving the cursor first is what makes the read legal:
    ///     `gatherRange` bounds at `absoluteOffset`, which after the write
    ///     is exactly `queryStart + chunk`. (Same range as post-write
    ///     `attendableViews()`, spelled out because it is the frozen region
    ///     that is wanted, not whatever `retainedCount` currently means.)
    ///  4. A frozen chunk never straddles M — `CBv2PrefixReusePlan
    ///     .clampedChunk` splits there and `write` traps if it did not — so
    ///     the range read is wholly cached.
    private func prefillKVWritingChunk(
        row: PagedSequenceKV, chunkKeys: MLXArray, chunkValues: MLXArray, dtype: DType
    ) -> PrefillKV {
        let queryStart = row.absoluteOffset
        let chunkLength = chunkKeys.dim(2)
        let rowKeys = chunkKeys.squeezed(axis: 0)
        let rowValues = chunkValues.squeezed(axis: 0)

        if row.frozenHighWater > queryStart {
            precondition(
                row.windowSize == nil,
                "[PagedLayerCache] layer \(layerIndex): a windowed row cannot be frozen")
            row.write(keys: rowKeys, values: rowValues)
            let start = row.baseOffset
            let (frozenKeys, frozenValues) = row.gatherRange(
                start: start, count: queryStart + chunkLength - start)
            return PrefillKV(
                keys: cast(frozenKeys, to: dtype), values: cast(frozenValues, to: dtype),
                start: start, queryStart: queryStart)
        }

        var start = row.baseOffset
        if let window = row.windowSize {
            start = max(start, queryStart - window + 1)
        }
        // The chunk lands in the slab under the POOL dtype, so attend the
        // values the slab will hold. Otherwise prefill would score these
        // tokens at a precision no later decode over them can reproduce,
        // which is exactly the kind of per-phase drift this file pins out.
        var chunkK = chunkKeys
        var chunkV = chunkValues
        if pool.config.dtype != chunkK.dtype {
            chunkK = chunkK.asType(pool.config.dtype)
            chunkV = chunkV.asType(pool.config.dtype)
        }
        guard start < queryStart else {
            row.write(keys: rowKeys, values: rowValues)
            return PrefillKV(
                keys: cast(chunkK, to: dtype), values: cast(chunkV, to: dtype),
                start: queryStart, queryStart: queryStart)
        }
        let (historyKeys, historyValues) = row.gatherRange(
            start: start, count: queryStart - start)
        row.write(keys: rowKeys, values: rowValues)
        return PrefillKV(
            keys: concatenated([cast(historyKeys, to: dtype), cast(chunkK, to: dtype)], axis: 2),
            values: concatenated(
                [cast(historyValues, to: dtype), cast(chunkV, to: dtype)], axis: 2),
            start: start, queryStart: queryStart)
    }

    @inline(__always)
    private func cast(_ array: MLXArray, to dtype: DType) -> MLXArray {
        array.dtype == dtype ? array : array.asType(dtype)
    }

    /// Per-request SDPA over the assembled chunk view, in QUERY BLOCKS
    /// (WS-0.2p).
    ///
    /// The score tensor is MATERIALIZED at `[1, queryHeads, L, kL]` on both
    /// the SDPA and the composed path, so an unblocked chunk holds a tensor
    /// linear in the chunk length AND the history. Blocking pins it at
    /// `[1, queryHeads, block, visible]` — that is what keeps the activation
    /// reserve independent of `prefillChunkSize` — and on a sliding layer it
    /// also drops the attention work that lies outside every query's window.
    ///
    /// Three things this loop deliberately does NOT do:
    ///  1. It does not call `CBv2AttentionV1.attendQueryBlocks`: that is
    ///     private, and its mask mode is symbolic `.causal`, which would
    ///     break this file's pinned "always `.array`" contract (MLX #3384).
    ///     Same block bounds, but the mask stays an absolute-position BOOL.
    ///  2. It does not gather per block. Consecutive blocks' visible spans
    ///     overlap by `window - 1` on a sliding layer and completely on a
    ///     full layer (`visibleStart == 0` for every block), so a per-block
    ///     gather is a 3-4x pessimisation. `prefillKVWritingChunk` gathers
    ///     ONCE and each block SLICES the result.
    ///  3. It does not rebuild the position vectors per block. They are host
    ///     `arange`s that get uploaded; rebuilding them in the loop repeats
    ///     nearly the whole history once per block on a full layer.
    ///
    /// The kill switch is `CBv2AttentionV1`'s, shared with the contiguous
    /// backend: `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK=0` restores the single
    /// unblocked call on BOTH backends. There is no paged-only knob.
    ///
    /// `sinks` arrives already prepared for the SDPA terminal (see
    /// `prefillSinks`) — this function and `attendQueryBlock` forward it
    /// unchanged rather than casting inside the block loop.
    private func prefillAttend(
        queries: MLXArray, kv: PrefillKV, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        let l = queries.dim(2)
        let historyCount = kv.historyCount
        precondition(
            kv.queryCount == l,
            "[PagedLayerCache] chunk view holds \(kv.queryCount) query columns, got \(l)")
        let k = cast(kv.keys, to: queries.dtype)
        let v = cast(kv.values, to: queries.dtype)
        let keyCount = k.dim(2)

        // Absolute positions: the queries are the trailing `l` key columns.
        // Built ONCE and sliced per block — because they are absolute, a
        // block's mask is exactly the corresponding slice of the whole
        // chunk's mask.
        let qStart = kv.queryStart
        let kStart = kv.start
        let qpos = MLXArray(Int32(qStart) ..< Int32(qStart + l)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(kStart) ..< Int32(kStart + keyCount)).expandedDimensions(
            axis: 0)

        // Span-bearing vision chunks (WS-2.2) keep the SINGLE unblocked
        // call. A block's key slice ends at the LATEST query's own position,
        // which is exactly the causal bound a bidirectional span is there to
        // escape: an image query must be able to attend keys AFTER itself
        // inside its own block, and those columns are not in the slice.
        // The overlay itself lives in `attendQueryBlock`, so the mask stays
        // defined in one place either way.
        guard boundSpanContext == nil, !kind.isBidirectional,
            CBv2AttentionV1.shouldBlockQueries(l)
        else {
            return attendQueryBlock(
                queries: queries, keys: k, values: v,
                qpos: qpos, kpos: kpos, queryStart: qStart,
                scale: scale, sinks: sinks)
        }

        // Block bounds are SHARED with the contiguous backend
        // (`CBv2AttentionV1.queryBlockBounds`), with this chunk's own
        // `historyCount`. That arithmetic is the activation-reserve bound;
        // two copies of it can drift into attending different spans.
        var window: Int?
        if case .slidingWindow(let w) = kind.attention { window = w }
        let blockSize = CBv2AttentionV1.queryBlockSize
        var outputs: [MLXArray] = []
        outputs.reserveCapacity((l + blockSize - 1) / blockSize)
        var offset = 0
        while offset < l {
            let count = min(blockSize, l - offset)
            let (visibleStart, visibleEnd) = CBv2AttentionV1.queryBlockBounds(
                historyCount: historyCount, offset: offset, count: count, window: window)
            outputs.append(
                attendQueryBlock(
                    queries: queries[0..., 0..., offset ..< (offset + count), 0...],
                    keys: k[0..., 0..., visibleStart ..< visibleEnd, 0...],
                    values: v[0..., 0..., visibleStart ..< visibleEnd, 0...],
                    qpos: qpos[offset ..< (offset + count), 0...],
                    kpos: kpos[0..., visibleStart ..< visibleEnd],
                    queryStart: qStart, scale: scale, sinks: sinks))
            offset += count
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }

    /// One query block. `qpos` (`[q, 1]`) and `kpos` (`[1, kL]`) are
    /// ABSOLUTE positions, so the causal-and-window BOOL mask is the same
    /// values the unblocked call would have produced for those rows and
    /// columns — the pinned `.array` representation never varies with the
    /// block size, and neither does the softcap path's.
    ///
    /// A bound span context (WS-2.2) OR-s a bidirectional-within-block
    /// overlay onto that base, matching MLXVLM Gemma4's
    /// `gemma4BidirectionalVisionMask` (`baseMask ∨ sameBlock`) and
    /// `CBv2AttentionV1.spanChunkMask` term for term. This is where paged
    /// composes MORE directly than contiguous: contiguous builds its base in
    /// RELATIVE coordinates (`boolMask(L:kL:window:)`) and has to
    /// reconstruct absolute `qAbs`/`kAbs` from `context.chunkEnd` to place
    /// the blocks; `qpos`/`kpos` are already absolute here, so the block
    /// bounds are compared directly against them and the two coordinate
    /// systems cannot drift apart.
    private func attendQueryBlock(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        qpos: MLXArray, kpos: MLXArray, queryStart: Int,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        var mask = kpos .<= qpos
        if case .slidingWindow(let window) = kind.attention {
            mask = mask & (kpos .> (qpos - Int32(window)))
        }
        if kind.isBidirectional {
            var reverse = (kpos .>= qpos) .&& (kpos .>= Int32(queryStart))
            if case .slidingWindow(let window) = kind.attention {
                reverse = reverse .&& (kpos .< (qpos + Int32(window)))
            }
            mask = mask .|| reverse
        }
        if let spans = boundSpanContext {
            for block in spans.blocks {
                let lo = Int32(block.tokenOffset)
                let hi = Int32(block.end)
                let qIn = (qpos .>= lo) .&& (qpos .< hi)  // [q, 1]
                let kIn = (kpos .>= lo) .&& (kpos .< hi)  // [1, kL]
                mask = mask .|| (qIn .&& kIn)
            }
        }

        if let softcap = attentionSoftcap {
            // SDPA cannot express logit softcapping — composed path, pinned
            // for softcap configs. It takes the same blocking, or a capped
            // config would keep the full unblocked peak.
            return PagedAttentionReference.composedAttention(
                queries: queries, keys: keys, values: values, scale: scale,
                boolMask: mask, sinks: sinks, softcap: softcap)
        }

        // `sinks` already promotes to the output dtype: the caller narrowed
        // it once per dispatch through `prefillSinks`. Do not re-cast here —
        // that is the per-block, per-row rebuild the hoist exists to remove.
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: .array(mask), sinks: sinks)
    }
}

// MARK: - Legacy KVCache conformance

/// Same shape as `CBv2LayerCache`'s conformance: lets the paged cache
/// travel through existing `[KVCache]` model plumbing (the models' v2
/// branches downcast to `CBv2AttendingLayerCache`); every legacy mutation
/// path TRAPS — v2-adapted models must call `updateAndAttend`.
extension PagedLayerCache: KVCache {
    /// Legacy scalar offset: max row offset. Host integers only — no sync.
    public var offset: Int {
        pagedRows.reduce(0) { max($0, $1.absoluteOffset) }
    }

    public var maxSize: Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// The paged slabs are pool-owned persistent buffers; per-step writes
    /// are materialized transitively by the engine's step asyncEval. The
    /// on-device `positionOffsets` advance chain is surfaced here (same
    /// convention as `CBv2LayerCache.innerState`) so it rides the step's
    /// eval set and cannot grow O(steps).
    public func innerState() -> [MLXArray] { [cachedPositionOffsets] }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "PagedLayerCache.update(keys:values:) is unsupported — v2-adapted models must call updateAndAttend (layer \(layerIndex))"
        )
    }

    public var state: [MLXArray] {
        get { [] }
        set {
            fatalError("PagedLayerCache has no serializable state (layer \(layerIndex))")
        }
    }

    public var metaState: [String] {
        get { [] }
        set {
            fatalError("PagedLayerCache has no metaState (layer \(layerIndex))")
        }
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        fatalError(
            "PagedLayerCache.makeMask is unsupported — v2 attention owns its masks (layer \(layerIndex))"
        )
    }

    public func copy() -> any KVCache {
        fatalError(
            "PagedLayerCache.copy is unsupported — v2 rows are engine-owned (layer \(layerIndex))")
    }
}

// MARK: - MTP rectangular verification

/// WS-3.4. `PagedLayerCache` honours the flag with a per-column loop over
/// the existing fused decode dispatch (`attendRectangularColumns`), so a
/// paged bank takes the rectangular verification path instead of the
/// `preconditionFailure` the engine used to hit on any cache that is not the
/// `final` `CBv2LayerCache`. See `PagedSeamContract.swift`.
extension PagedLayerCache: CBv2MTPRectangularSerializing {}

// MARK: - KV-borrow chunk retention

/// WS-1.2. A KV-shared sibling attends this layer's assembled chunk view,
/// and the shrunk ring cannot reproduce the older part of it once the chunk
/// is written, so the source has to keep it. Retention is ON by default (a
/// cache used without a bank stays correct) and `CBv2LayerCacheBank` turns
/// it off for every layer no sibling borrows — all of them for gemma-4
/// (`num_kv_shared_layers: 0`) and gpt-oss, so neither model pays for it.
extension PagedLayerCache: CBv2KVSourceChunkRetaining {
    public func setRetainsChunkForBorrowers(_ retains: Bool) {
        retainsChunkForBorrowers = retains
        if !retains { retainedPrefillKV = [] }
    }
}

// MARK: - Packed prefill (WS-2.1)

/// The affirmative claim `CBv2LayerCacheBank.supportsPackedPrefill` reads.
///
/// `updateAndAttend`'s prompt-chunk branch loops over the bound rows and
/// runs each one through the SAME `prefillKVWritingChunk` / `prefillAttend`
/// sequence a `[1, chunk]` call runs, then concatenates on the batch axis.
/// No tensor in that sequence spans two rows: the gather walks one row's
/// page table, the write lands in one row's pages, and the mask is built
/// from one row's absolute positions. A packed row is therefore
/// bit-identical to that row run alone, which is what this claim asserts.
extension PagedLayerCache: CBv2PackedPrefillCapableCache {
    /// Type-level form of the same claim, for callers that must decide
    /// BEFORE any pool or cache exists — see
    /// `honorsSpanMaskContextsByConstruction` for why that matters.
    public static let keepsRowsIndependentWhenPackedByConstruction = true

    public var keepsRowsIndependentWhenPacked: Bool {
        Self.keepsRowsIndependentWhenPackedByConstruction
    }
}

// MARK: - Vision span masks (WS-2.2)

/// The affirmative claim `CBv2LayerCacheBank.supportsMultimodalSpans` reads,
/// plus the binding it refines.
///
/// The bound context reaches the mask: `prefillAttend` forces the single
/// unblocked call while it is set, and `attendQueryBlock` OR-s the
/// bidirectional-within-block overlay onto the absolute-coordinate base
/// mask. Both are exercised by `CBv2PagedPackedSpanTests`.
extension PagedLayerCache: CBv2MultimodalSpanCapableCache {
    /// The SAME claim, readable without an instance.
    ///
    /// The provider's slot policy has to decide whether a VLM slot may
    /// route to paged before it builds anything, so it cannot ask a live
    /// cache. It could hardcode `true` on its side; that is precisely the
    /// "capability ASSUMED rather than asked" defect, because the assumption
    /// would then live in a different repository from the implementation and
    /// would not move when this file does.
    ///
    /// Instead there is exactly ONE boolean. The engine's own submit-time
    /// gate (`CBv2LayerCacheBank.supportsMultimodalSpans`) reads the instance
    /// property below, which returns this constant; the provider's routing
    /// gate reads this constant directly. Regress the overlay and both flip
    /// together — production stops routing vision here in the same edit that
    /// stops honouring it.
    ///
    /// The residual gap a type-level claim cannot close is that the constant
    /// is not derived from the mask code. That is what
    /// `CBv2PagedPackedSpanTests.typeLevelClaimsMatchARealPagedBank` and the
    /// span-behaviour tests are for: the first pins the constant to what a
    /// real bank of real caches answers, the rest pin the answer to the
    /// numbers.
    public static let honorsSpanMaskContextsByConstruction = true

    public func bindSpanContext(_ context: CBv2SpanChunkContext?) {
        boundSpanContext = context
    }

    public var honorsSpanMaskContexts: Bool {
        Self.honorsSpanMaskContextsByConstruction
    }
}
