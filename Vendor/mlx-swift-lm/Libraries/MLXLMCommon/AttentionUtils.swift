import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
///
/// LIMITATION (ContinuousBatchingV2 caches): when `cache` is a
/// `CBv2AttendingLayerCache`, the layer cache owns BOTH the KV update and
/// the attention computation, INCLUDING masking — the `mask` parameter is
/// DISCARDED on that path (v2 derives causal/window masks from per-row
/// absolute positions), and `sinks` are passed as nil. A non-adapted model
/// driven with CBv2 caches therefore silently loses any CUSTOM array mask
/// (e.g. bidirectional/prefix-LM or padding masks) and any attention
/// sinks; sinks-bearing or custom-mask models must call `updateAndAttend`
/// directly instead. Array masks fail HARD here — in release builds too
/// (`preconditionFailure`, not a debug-only assertion): silently swapping
/// the model's required mask for v2's causal/window mask would corrupt
/// output in exactly the builds users run (PR#62 review).
///
/// MULTI-ROW LIMITATION: this compatibility path is B == 1 ONLY. Legacy
/// (non-v2-adapted) models apply scalar RoPE via `KVCache.offset` BEFORE
/// calling this helper; a CBv2 cache's legacy `offset` is the MAX row
/// offset, so at B > 1 every shorter row would be silently mis-rotated.
/// B == 1 stays allowed (the scalar offset is exact for a single row).
/// Generic models must be v2-adapted — capture `positionOffsets` before
/// dispatch and call `updateAndAttend` directly — before they can serve
/// multi-row CBv2 batches. This fails loudly rather than mis-rotating.
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    // ContinuousBatchingV2 hook — see the LIMITATION notes above.
    if let v2 = cache as? CBv2AttendingLayerCache {
        if let violation = cbv2CustomMaskViolation(mask: mask, layerIndex: v2.layerIndex) {
            preconditionFailure(violation)
        }
        if let violation = cbv2LegacyAttentionBatchViolation(
            batch: queries.dim(0), layerIndex: v2.layerIndex)
        {
            preconditionFailure(violation)
        }
        return v2.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
    }
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        // WIDE-DECODE EXACTNESS CHUNK (B == 1, causal, 6 <= qL <= 9): the
        // fused sdpa vector path serves qL * gqa <= 32; above it the dispatch
        // changes kernel family and the accumulation order of every score —
        // the measured source of the MTP width wall's top-2 VALUE drift.
        // Splitting the queries at row 5 keeps both halves on the fused
        // vector path with windows that are BYTE-IDENTICAL to two
        // consecutive <= 5-row rounds at the same offsets: with bottom-right
        // causal alignment, chunk A (rows 0..<5) over keys[..<kL-(qL-5)]
        // gives row i the window a width-5 round would, and chunk B
        // (rows 5..) over the full keys gives row 5+j the window a follow-up
        // width-(qL-5) round would. Keys/values are re-sliced, not
        // recomputed — the only extra cost is one more pass over the KV
        // rows (a few MB), never over weights. Serial (qL == 1), the <= 5
        // verify widths, and prefill (qL > 9) are untouched.
        // The cache update above happens exactly once; both segments below
        // are read-only views of that single committed candidate window.
        let qL = queries.dim(2)
        let kL = cachedKeys.dim(2)
        if queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL,
           case .causal = mask
        {
            let split = 5
            let kSplit = kL - (qL - split)
            let outA = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., 0 ..< split, 0...],
                keys: cachedKeys[0..., 0..., 0 ..< kSplit, 0...],
                values: cachedValues[0..., 0..., 0 ..< kSplit, 0...],
                scale: scale,
                mask: .causal
            )
            let outB = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., split..., 0...],
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: .causal
            )
            return concatenated([outA, outB], axis: 2)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

/// Custom-mask guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the LIMITATION doc there). `.none`/`.causal` are subsumed by v2's own
/// position-derived masks; any other mode (custom `.array`/`.arrays`) would
/// be silently DISCARDED by `updateAndAttend`, so it must fail in ALL build
/// configurations — a debug-only `assertionFailure` compiles out of release
/// builds and lets the wrong mask ship (PR#62 review). Returns the failure
/// description for an illegal call, or nil when the call is allowed.
/// Internal (not private) so tests can pin the exact condition without
/// tripping the precondition.
func cbv2CustomMaskViolation(
    mask: MLXFast.ScaledDotProductAttentionMaskMode, layerIndex: Int
) -> String? {
    switch mask {
    case .none, .causal:
        return nil
    default:
        return """
            attentionWithCacheUpdate: a custom array mask was passed with a \
            CBv2 layer cache (layer \(layerIndex)). CBv2 caches own their \
            masks and DISCARD this parameter — the model must be v2-adapted \
            (call updateAndAttend with its own semantics) instead.
            """
    }
}

/// Multi-row guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the MULTI-ROW LIMITATION doc there). Returns the failure description for
/// an illegal call, or nil when the call is allowed. Internal (not private)
/// so tests can pin the exact condition without tripping the precondition.
func cbv2LegacyAttentionBatchViolation(batch: Int, layerIndex: Int) -> String? {
    guard batch > 1 else { return nil }
    return """
        attentionWithCacheUpdate: a multi-row batch (B=\(batch)) reached the \
        legacy CBv2 compatibility path (layer \(layerIndex)). Legacy models \
        apply scalar RoPE via `KVCache.offset` — the MAX row offset — so \
        shorter rows would be silently mis-rotated at B > 1. This model must \
        be v2-adapted (read `positionOffsets` before dispatch and call \
        `updateAndAttend` directly) before multi-row CBv2 serving; B == 1 \
        remains supported.
        """
}

// MARK: - Drafter per-row Q-only SDPA (B > 1, view-backed KV)
//
// The MTP drafter borrows the target's per-row contiguous KV (CBv2 v1
// backend: `CBv2FullSequenceKV` / `CBv2WindowedSequenceKV`). The legacy
// drafter path materializes those per-row strided views into one
// padded/stacked `[B, kvHeads, Tmax, headDim]` tensor before each drafter
// Q-only SDPA — an extra HBM round-trip the model didn't have to make,
// because each per-row tensor is already a strided view of the contiguous
// KV leg. The helpers below attend per row and concatenate the outputs,
// so the drafter's Q-only SDPA reads the target's K/V directly from the
// pinned contiguous storage without an intermediate materialization.

/// Q-only SDPA for a B > 1 drafter that already owns per-row K/V strided
/// views of the target's contiguous KV leg. Each row's `(keys, values)` is
/// `[1, kvHeads, T_row, headDim]` in temporal order; `queries` is the
/// drafter's per-row Q `[B, nHeads, qL, headDim]`. Rows are attended one
/// at a time and the outputs concatenated on the batch axis. No K/V
/// materialization, no host sync.
public func drafterPerRowQOnlySDPA(
    queries: MLXArray,
    perRowKV: [(keys: MLXArray, values: MLXArray)],
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    precondition(
        queries.dim(0) == perRowKV.count,
        "drafterPerRowQOnlySDPA: query batch \(queries.dim(0)) != KV row count \(perRowKV.count)")
    precondition(!perRowKV.isEmpty, "drafterPerRowQOnlySDPA: empty per-row KV")
    var parts: [MLXArray] = []
    parts.reserveCapacity(perRowKV.count)
    for (rowIndex, kv) in perRowKV.enumerated() {
        let rowQ = queries[rowIndex ..< rowIndex + 1, 0..., 0..., 0...]
        let attended = MLXFast.scaledDotProductAttention(
            queries: rowQ,
            keys: kv.keys,
            values: kv.values,
            scale: scale,
            mask: mask)
        parts.append(attended)
    }
    return parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
}

/// Variant of `drafterPerRowQOnlySDPA` that accepts one pre-sliced
/// `MLXArray` view per row (the encoder-side fusion contract: the drafter
/// build can construct each view with a stride-friendly slice, never a
/// materializing copy). The KV pair is the single contiguous tensor pair
/// that backs the B > 1 drafter's frozen capture; per-row strides are
/// already encoded in each `MLXArray` so SDPA reads them in place.
public func drafterViewPerRowQOnlySDPA(
    queries: MLXArray,
    perRowKeyViews: [MLXArray],
    perRowValueViews: [MLXArray],
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    precondition(
        queries.dim(0) == perRowKeyViews.count
            && perRowKeyViews.count == perRowValueViews.count,
        "drafterViewPerRowQOnlySDPA: query batch must match per-row KV count")
    precondition(!perRowKeyViews.isEmpty, "drafterViewPerRowQOnlySDPA: empty per-row KV")
    var parts: [MLXArray] = []
    parts.reserveCapacity(perRowKeyViews.count)
    for (rowIndex, _) in perRowKeyViews.enumerated() {
        let rowQ = queries[rowIndex ..< rowIndex + 1, 0..., 0..., 0...]
        let attended = MLXFast.scaledDotProductAttention(
            queries: rowQ,
            keys: perRowKeyViews[rowIndex],
            values: perRowValueViews[rowIndex],
            scale: scale,
            mask: mask)
        parts.append(attended)
    }
    return parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
}
