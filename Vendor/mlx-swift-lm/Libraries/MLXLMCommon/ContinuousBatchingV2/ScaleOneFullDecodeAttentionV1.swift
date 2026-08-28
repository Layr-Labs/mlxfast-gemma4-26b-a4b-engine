// ScaleOneFullDecodeAttentionV1.swift
//
// Exact MLX SDPA fallback graph for Gemma 4 full-attention decode when the
// query scale is 1.0. The vendored fallback begins by materializing
// `BF16(1.0) * queries`; multiplying BF16 by exactly one is bit-identical, so
// the physical query producer can feed the unchanged GQA graph directly.

import Foundation
import MLX

enum CBv2ScaleOneFullDecodeAttentionV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_SCALE_ONE_FULL_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Reproduce `mlx::core::fast::scaled_dot_product_attention`'s D=512
    /// fallback after its leading scale multiply. Every operation below maps
    /// one-for-one to fast.cpp:724-788: GQA unflatten/expand, QK matmul,
    /// precise softmax, scores-V matmul, and head flatten.
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, queryLength: Int, keyLength: Int,
        window: Int?, sinks: MLXArray?, bidirectional: Bool
    ) -> MLXArray? {
        let batch = queries.dim(0)
        let queryStrides = queries.strides
        guard enabled,
            batch == 1 || batch == 8,
            scale == 1.0,
            queryLength == 1,
            keyLength > 0,
            window == nil,
            sinks == nil,
            !bidirectional,
            queries.dtype == .bfloat16,
            keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            queries.shape == [batch, 16, 1, 512],
            keys.shape == [batch, 2, keyLength, 512],
            values.shape == keys.shape,
            // Gemma's `[B, L, H, D] -> [B, H, L, D]` decode transpose is
            // physically linear even though singleton L retains its old
            // stride. These three strides prove the B/H/D address order.
            queryStrides.count == 4,
            queryStrides[0] == 16 * 512,
            queryStrides[1] == 512,
            queryStrides[3] == 1
        else { return nil }

        // `Unflatten` sees a row-contiguous physical view (singleton strides
        // are ignored by MLX's row-contiguous flag) and publishes canonical
        // output strides without copying. This gives the following M=1 matmul
        // the same Q layout the removed elementwise multiply produced.
        let q = unflatten(queries, axis: 1, shape: [2, 8])
        let k = expandedDimensions(keys, axis: 2)
        let v = expandedDimensions(values, axis: 2)
        let scores = matmul(q, k.swappedAxes(-1, -2))
        let probabilities = softmax(scores, axis: -1, precise: true)
        let groupedOutput = matmul(probabilities, v)

        CBv2EngageMark.once("scale-one-full-attn")
        return flatten(groupedOutput, startAxis: 1, endAxis: 2)
    }
}
