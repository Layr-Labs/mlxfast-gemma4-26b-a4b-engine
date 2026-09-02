import Foundation
import MLX

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
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
