// LayerCacheV2.swift
//
// The per-layer batch-facing cache object v2 models interact with.
//
// `CBv2LayerCache` conforms to BOTH:
//  - `CBv2AttendingLayerCache` — the v2 surface: `updateAndAttend` owns the
//    KV update AND the attention computation (backends are swappable), and
//  - the legacy `KVCache` protocol — so it can travel through existing
//    `[KVCache]` plumbing. `update(keys:values:)` TRAPS: v2-adapted models
//    must call `updateAndAttend` (via the hook in `attentionWithCacheUpdate`).
//
// Batch membership is object membership: join = `appendRow`, leave =
// `removeRow`. There is no shared frontier, no left padding, no batch-wide
// trim — a row's KV state and positions cannot be affected by its batchmates.

import Foundation
import MLX

/// Shared on-device position chain for one contiguous cache bank. The bank
/// chooses one owning cache to rebuild/advance it; every cache reads the same
/// value, so the model can snapshot it once before entering the layer loop.
final class CBv2PositionOffsetsState {
    var value: MLXArray

    init(rows: [CBv2SequenceKV]) {
        value = MLXArray(rows.map { Int32($0.absoluteOffset) })
    }

    func rebuild(from rows: [CBv2SequenceKV]) {
        value = MLXArray(rows.map { Int32($0.absoluteOffset) })
    }
}

/// Per-layer device fence ordering the fused decode ring write.
///
/// The fused ring pass A stores this step's K/V into the retained ring
/// allocation IN PLACE — a side effect the array graph cannot otherwise see.
/// Threading a one-element int32 through the kernel (in as `write_fence`, out
/// as `fence`) makes it a real data dependency: the next step's writing pass A
/// consumes the fence this step produced, so the in-place store is part of the
/// evaluated chain instead of relying on host timing. `innerState()` publishes
/// the value so the loop's per-step `asyncEval` collapses that chain, exactly
/// as it does for the position-offset chain.
final class CBv2DecodeRingWriteFence {
    var value = MLXArray.zeros([1], dtype: .int32)
}

/// Per-layer, batch-facing cache + attention dispatcher for the v2 engine.
public final class CBv2LayerCache: CBv2AttendingLayerCache {

    public let layerIndex: Int
    public let kind: CBv2LayerKind

    /// Ordered per-row sequence states (row order == batch row order).
    /// Empty for KV-shared layers (`kind.sharesKVWithLayer != nil`), which
    /// own no storage and borrow via `attendBorrowing`.
    public private(set) var rows: [CBv2SequenceKV]

    /// Per-row absolute RoPE offsets `[B]` (int32, device array).
    ///
    /// REBUILT from host integers only on membership changes; ADVANCED
    /// on-device (`+ L`) inside `updateAndAttend`. The step loop therefore
    /// never uploads fresh host arrays and never syncs (`.item()`) — the
    /// engine loop's per-step `asyncEval` (over `innerState()`) collapses
    /// the lazy `+ L` chain so it cannot grow O(steps) (DAR-325).
    ///
    /// NOTE: models must read this BEFORE calling `updateAndAttend` for the
    /// step (it holds the offsets of the tokens about to be processed), and
    /// KV-shared layers must reuse the SOURCE layer's pre-update capture —
    /// the same discipline as `gemma4CapturePositionOffset`.
    public var positionOffsets: MLXArray { positionOffsetsState.value }

    /// Non-nil only after a cache bank has unified every contiguous layer on
    /// one position chain. This explicit capability keeps standalone and paged
    /// cache semantics unchanged.
    public var unifiedPositionOffsets: MLXArray? {
        usesUnifiedPositionOffsets ? positionOffsetsState.value : nil
    }

    /// Stable identity of the bank-owned shared position state. Comparing
    /// the MLXArray values themselves would evaluate them; the adapter uses
    /// this host-only identity to prove every layer shares one chain.
    var unifiedPositionStateIdentity: ObjectIdentifier? {
        usesUnifiedPositionOffsets ? ObjectIdentifier(positionOffsetsState) : nil
    }

    /// Explicit ordering root for the fused in-place sliding-ring write.
    /// Kept in the initial conservative compaction even though the logits
    /// graph also reaches the multi-output primitive that produces it.
    var decodeRingWriteFenceEvaluationRoot: MLXArray { decodeRingWriteFence.value }

    private var positionOffsetsState: CBv2PositionOffsetsState
    private var usesUnifiedPositionOffsets = false
    private var advancesPositionOffsets = true
    private let decodeRingWriteFence = CBv2DecodeRingWriteFence()

    /// Whether a KV-shared sibling may still be attending views of this
    /// layer's storage. `CBv2LayerCacheBank` clears it for every layer nothing
    /// borrows (see `CBv2KVSourceChunkRetaining`); while it is set, the fused
    /// in-place ring write is refused and decode keeps the copying
    /// `SliceUpdate` path, so a borrower can never observe a mutated buffer.
    private var retainsChunkForBorrowers = true

    /// MTP-only verification policy. When true, an L>1 update still projects
    /// and stores the whole rectangle once, but attention evaluates each
    /// query with the canonical L=1 SDPA path and its exact visible KV prefix.
    var mtpSerializesRectangularAttention = false

    /// Times `positionOffsets` was rebuilt from host integers. Tests assert
    /// this only moves on membership changes — never inside the step loop.
    public private(set) var positionOffsetsHostRebuilds = 0

    /// Optional attention-logit soft cap (`cap * tanh(qk / cap)` before
    /// softmax, Gemma-2 style). Construction-time configuration from model
    /// config — identical plumbing on both backends (`PagedLayerCache` takes
    /// the same parameter); never part of the per-call contract surface.
    public let attentionSoftcap: Float?

    /// Optional vision span context for each CURRENT prefill row. The engine
    /// binds this array immediately before graph construction and clears it
    /// immediately after. nil outside that window; nil entries are ordinary
    /// text rows sharing a rectangular call.
    private(set) var boundSpanContexts: [CBv2SpanChunkContext?]?

    public init(
        layerIndex: Int, kind: CBv2LayerKind, rows: [CBv2SequenceKV] = [],
        attentionSoftcap: Float? = nil
    ) {
        precondition(
            kind.sharesKVWithLayer == nil || rows.isEmpty,
            "CBv2LayerCache: KV-shared layers own no rows")
        self.layerIndex = layerIndex
        self.kind = kind
        self.rows = rows
        self.attentionSoftcap = attentionSoftcap
        self.positionOffsetsState = CBv2PositionOffsetsState(rows: rows)
    }

    /// Bank-only wiring performed before rows are bound. Exactly one owning
    /// cache advances the shared chain; all other caches expose it read-only.
    func unifyPositionOffsets(
        with state: CBv2PositionOffsetsState, advances: Bool
    ) {
        positionOffsetsState = state
        usesUnifiedPositionOffsets = true
        advancesPositionOffsets = advances
    }

    // MARK: - Membership (the ONLY places positionOffsets is host-rebuilt)

    public func appendRow(_ row: CBv2SequenceKV) {
        precondition(
            kind.sharesKVWithLayer == nil, "CBv2LayerCache: cannot add rows to a KV-shared layer")
        rows.append(row)
        rebuildPositionOffsets()
    }

    public func removeRow(at index: Int) {
        rows.remove(at: index)
        rebuildPositionOffsets()
    }

    /// Replace the whole row set (batch recomposition). Also the correct way
    /// to re-sync `positionOffsets` after out-of-band row mutation
    /// (e.g. rollback during speculative verification).
    public func setRows(_ newRows: [CBv2SequenceKV]) {
        setRows(newRows, rebuildPositionOffsets: true)
    }

    /// Bank path: non-canonical unified caches update row bindings without
    /// rebuilding the one shared host-derived position tensor.
    func setRows(
        _ newRows: [CBv2SequenceKV], rebuildPositionOffsets shouldRebuild: Bool
    ) {
        precondition(
            kind.sharesKVWithLayer == nil || newRows.isEmpty,
            "CBv2LayerCache: KV-shared layers own no rows")
        rows = newRows
        if shouldRebuild { rebuildPositionOffsets() }
    }

    // MARK: - CBv2AttendingLayerCache

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer == nil,
            "CBv2LayerCache: KV-shared layer \(layerIndex) must use attendBorrowing")
        let output = CBv2AttentionV1.updateAndAttend(
            rows: rows, kind: kind,
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: sinks, softcap: attentionSoftcap,
            spanContexts: boundSpanContexts,
            serializeQueries: mtpSerializesRectangularAttention,
            decodeRingWriteFence: decodeRingWriteFence,
            allowFusedRingWrite: !retainsChunkForBorrowers)
        // Advance offsets ON-DEVICE. A unified bank elects exactly one owning
        // cache; Gemma snapshots the shared pre-step value before this call.
        if advancesPositionOffsets {
            positionOffsetsState.value = positionOffsetsState.value + Int32(queries.dim(2))
        }
        return output
    }

    /// MTP widened-round stash: the whole round's staged K/V and packed
    /// stripe for this layer, consumed by `mtpCommitWidened`.
    public private(set) var mtpWidenedStash:
        (keys: MLXArray, values: MLXArray, stripe: MLXArray,
         columns: Int, bases: [Int])?

    /// MTP-BATCHED-VERIFY widened road: ONE grouped stripe pack, counters-only
    /// staging, ONE multi-column attend. All admissions are checked (and the
    /// stripe built) BEFORE any counter moves; a kernel refusal after staging
    /// reverts the counters and returns nil so the caller's fallback starts
    /// clean.
    public func mtpVerifyStageAndAttendColumnsWidened(
        queries: MLXArray,
        keysAll: MLXArray,
        valuesAll: MLXArray,
        columns: Int,
        scale: Float
    ) -> MLXArray? {
        // Refusal is an ordinary, exact fallback (not an error). Keep it
        // silent: full-attention layers intentionally try this sliding arm
        // before the D512 arm, so synchronous stderr writes here would sit in
        // every layer/round hot path.
        func refuse(_: String) -> MLXArray? { nil }
        guard kind.sharesKVWithLayer == nil else { return refuse("shared") }
        guard columns >= 2, columns <= 8 else { return refuse("columns") }
        guard attentionSoftcap == nil else { return refuse("softcap") }
        guard boundSpanContexts == nil else { return refuse("span") }
        guard scale == 1.0 else { return refuse("scale=\(scale)") }
        guard queries.ndim == 4,
            queries.dim(0) == rows.count * columns,
            queries.dim(2) == 1
        else { return refuse("queries=\(queries.shape) rows=\(rows.count)") }
        guard mtpWidenedStash == nil else { return refuse("stash-live") }
        let ringRows = rows.compactMap { $0 as? CBv2WindowedSequenceKV }
        guard ringRows.count == rows.count else { return refuse("rows-kind") }
        guard ringRows.allSatisfy({ $0.verifyStripeEligible })
        else { return refuse("eligible") }
        guard ringRows.allSatisfy({ $0.window == queriesWindowPin })
        else { return refuse("window") }
        guard ringRows.allSatisfy({ $0.verifyCommitBuffers != nil })
        else { return refuse("buffers") }
        guard
            let stripe = CBv2RaggedTwoPassDecodeAttentionV1.packVerifyStripeGrouped(
                keysAll: keysAll, valuesAll: valuesAll, columns: columns)
        else {
            return refuse(
                "pack keys=\(keysAll.shape) values=\(valuesAll.shape) cols=\(columns)")
        }

        let bases = ringRows.map { $0.absoluteOffset }
        for row in ringRows { row.stageWidenedCountersOnly(columns) }
        let starts = ringRows.map {
            ($0.absoluteOffset - (columns - 1)) % queriesWindowPin
        }
        var mirrors: [MLXArray] = []
        for row in ringRows {
            guard let buffers = row.verifyCommitBuffers else {
                for staged in ringRows where staged.widenedStagedCount > 0 {
                    staged.revertWidenedStage()
                }
                return nil
            }
            mirrors.append(buffers.mirror)
        }
        guard let output = CBv2RaggedTwoPassDecodeAttentionV1.attendRingQuantVerifyMC(
            queries: queries, mirrors: mirrors, stripe: stripe,
            starts: starts, columns: columns, scale: scale,
            slidingWindowLength: queriesWindowPin,
            orderingFence: decodeRingWriteFence.value)
        else {
            for staged in ringRows where staged.widenedStagedCount > 0 {
                staged.revertWidenedStage()
            }
            return refuse("mc-kernel q=\(queries.shape) starts=\(starts)")
        }
        mtpWidenedStash = (keysAll, valuesAll, stripe, columns, bases)
        if advancesPositionOffsets {
            positionOffsetsState.value = positionOffsetsState.value + Int32(columns)
        }
        return output
    }

    /// MTP-BATCHED-VERIFY, D512 full-attention arm: ONE widened store then
    /// per-column chains fenced only on it. nil BEFORE side effects.
    public func mtpVerifyAttendColumnsD512(
        queriesAll: MLXArray,
        keysAll: MLXArray,
        valuesAll: MLXArray,
        columns: Int,
        scale: Float
    ) -> MLXArray? {
        guard kind.sharesKVWithLayer == nil,
            attentionSoftcap == nil,
            boundSpanContexts == nil
        else { return nil }
        guard
            let fused = CBv2RaggedComposedD512DecodeAttentionV1
                .updateAndAttendVerifyColumns(
                    rows: rows, kind: kind,
                    queriesAll: queriesAll, keysAll: keysAll,
                    valuesAll: valuesAll, columns: columns,
                    previousWriteFence: decodeRingWriteFence.value,
                    scale: scale)
        else { return nil }
        decodeRingWriteFence.value = fused.nextWriteFence
        if advancesPositionOffsets {
            positionOffsetsState.value = positionOffsetsState.value + Int32(columns)
        }
        return fused.output
    }

    /// The pinned sliding geometry the exact verify kernels serve.
    private var queriesWindowPin: Int { 1024 }

    /// Commit the widened round's confirmed prefixes through the widened
    /// commit kernel (ring BF16 + mirror q4 in place, fence-chained).
    /// `confirmed[r]` counts row r's committed columns; 0 writes nothing.
    public func mtpCommitWidened(confirmed: [Int]) {
        guard let stash = mtpWidenedStash else { return }
        mtpWidenedStash = nil
        let ringRows = rows.compactMap { $0 as? CBv2WindowedSequenceKV }
        guard ringRows.count == rows.count, confirmed.count == ringRows.count
        else { return }
        var ringKeys: [MLXArray] = []
        var ringValues: [MLXArray] = []
        var mirrors: [MLXArray] = []
        for row in ringRows {
            guard let buffers = row.verifyCommitBuffers else { return }
            ringKeys.append(buffers.ringKeys)
            ringValues.append(buffers.ringValues)
            mirrors.append(buffers.mirror)
        }
        if let fence = CBv2RaggedTwoPassDecodeAttentionV1.commitVerifyWidened(
            keysAll: stash.keys, valuesAll: stash.values, stripe: stash.stripe,
            confirmed: confirmed, bases: stash.bases, columns: stash.columns,
            ringKeys: ringKeys, ringValues: ringValues, mirrors: mirrors,
            previousWriteFence: decodeRingWriteFence.value)
        {
            decodeRingWriteFence.value = fence
        }
    }

    /// Final-layer prompt specialization (see LastQueryPrefillV2.swift):
    /// commit the whole chunk's K/V, attend only its newest query row.
    /// Offsets advance by the K/V length, NOT the query length — the chunk
    /// consumed `keys.dim(2)` positions even though one query was evaluated.
    public func updateAndAttendLastQuery(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer == nil,
            "CBv2LayerCache: KV-shared layer \(layerIndex) owns no storage to commit")
        precondition(
            !mtpSerializesRectangularAttention,
            "CBv2LayerCache: last-query prefill is never part of an MTP verify round")
        let output = CBv2AttentionV1.updateAndAttendLastQuery(
            rows: rows, kind: kind,
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: sinks, softcap: attentionSoftcap)
        if advancesPositionOffsets {
            positionOffsetsState.value = positionOffsetsState.value + Int32(keys.dim(2))
        }
        return output
    }

    public func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer != nil,
            "CBv2LayerCache: attendBorrowing called on a storage-owning layer")
        precondition(
            kind.sharesKVWithLayer == source.layerIndex,
            "CBv2LayerCache: layer \(layerIndex) shares KV with \(kind.sharesKVWithLayer!), not \(source.layerIndex)"
        )
        return CBv2AttentionV1.attendBorrowing(
            sourceRows: source.rows, sourceKind: source.kind, kind: kind,
            queries: queries, scale: scale, sinks: sinks, softcap: attentionSoftcap,
            spanContexts: boundSpanContexts,
            serializeQueries: mtpSerializesRectangularAttention)
    }

    // MARK: - Private

    private func rebuildPositionOffsets() {
        positionOffsetsHostRebuilds += 1
        CBv2CoreInstrumentation.recordPositionOffsetsHostRebuild()
        positionOffsetsState.rebuild(from: rows)
    }
}

// MARK: - Borrower retention (fused ring-write eligibility)

extension CBv2LayerCache: CBv2KVSourceChunkRetaining {
    /// The bank owns the borrower map, so it is the only thing that can tell
    /// a source layer whether anything borrows from it. Cleared here means
    /// "no sibling attends this layer's buffers", which is what makes an
    /// in-place decode ring write safe.
    public func setRetainsChunkForBorrowers(_ retains: Bool) {
        retainsChunkForBorrowers = retains
    }
}

// MARK: - Final-layer last-query prefill

extension CBv2LayerCache: CBv2LastQueryPrefillLayerCache {}

// MARK: - Vision span-mask binding

extension CBv2LayerCache: CBv2PackedSpanMaskBinding {
    public func bindSpanContext(_ context: CBv2SpanChunkContext?) {
        boundSpanContexts = context.map { [$0] }
    }

    public func bindSpanContexts(_ contexts: [CBv2SpanChunkContext?]?) {
        boundSpanContexts = contexts
    }
}

// MARK: - Legacy KVCache conformance

extension CBv2LayerCache: KVCache {
    /// Legacy scalar offset: max row offset. Host integers only — no sync.
    public var offset: Int {
        rows.reduce(0) { max($0, $1.absoluteOffset) }
    }

    public var maxSize: Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// The engine loop evaluates cache inner state each step (asyncEval) to
    /// collapse lazy chains: per-row storage plus the positionOffsets chain.
    public func innerState() -> [MLXArray] {
        var arrays = [positionOffsetsState.value, decodeRingWriteFence.value]
        for row in rows {
            if let provider = row as? CBv2InnerStateProviding {
                arrays.append(contentsOf: provider.cbv2InnerState())
            }
        }
        return arrays
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "CBv2LayerCache.update(keys:values:) is unsupported — v2-adapted models must call updateAndAttend (layer \(layerIndex))"
        )
    }

    public var state: [MLXArray] {
        get { [] }
        set {
            fatalError("CBv2LayerCache has no serializable state (layer \(layerIndex))")
        }
    }

    public var metaState: [String] {
        get { [] }
        set {
            fatalError("CBv2LayerCache has no metaState (layer \(layerIndex))")
        }
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        fatalError(
            "CBv2LayerCache.makeMask is unsupported — v2 attention owns its masks (layer \(layerIndex))"
        )
    }

    public func copy() -> any KVCache {
        fatalError(
            "CBv2LayerCache.copy is unsupported — v2 rows are engine-owned (layer \(layerIndex))")
    }
}
