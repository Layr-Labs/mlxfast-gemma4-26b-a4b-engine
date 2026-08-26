// Copyright © 2026 Apple Inc.
//
// Bidirectional attention-mask helpers for the Gemma 4 MTP drafter.
//
// The drafter attends from L query positions (placed just past the end of
// the target's KV cache) over the target's last-layer K/V — bidirectionally
// for both full-attention and sliding-window layers. Full attention is
// always `.none` (SDPA treats it as fully attending). SWA short-circuits
// to `.none` when every query position's window covers the whole KV; that
// is the regime `RotatingKVCache` always produces (kv_len ≤ sliding_window).
// Only when the KV is longer than the window do we materialize an additive
// `-inf`/`0` bias.
//
// Reference: mlx_vlm/speculative/drafters/gemma4_assistant/masks.py in
// Blaizzy/mlx-vlm#1112 (merged 244f4bb).

import Foundation
import MLX
import MLXNN

public enum DrafterMasks {

    /// Full-attention mask: always `.none`. SDPA handles bidirectional
    /// attention without an explicit mask.
    public static func bidirectionalFull(
        queryLen: Int, kvLen: Int, dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        _ = (queryLen, kvLen, dtype)
        return .none
    }

    /// Bidirectional sliding-window mask.
    ///
    /// For each query position `q ∈ [queryOffset, queryOffset + queryLen)`,
    /// allow attention to KV positions `k ∈ (q - window, q + window)`.
    /// Returns `.none` when the whole KV fits inside every query's window.
    /// Otherwise returns a materialized `.array(...)` additive mask with
    /// `-inf` outside the window and `0` inside.
    public static func bidirectionalSWA(
        queryLen: Int, queryOffset: Int, kvLen: Int,
        window: Int, dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // No masking needed: the entire KV fits in the bidirectional window
        // of every query.
        if kvLen <= window && queryOffset + queryLen <= kvLen + window {
            return .none
        }

        let qIdx = MLXArray(Int32(queryOffset) ..< Int32(queryOffset + queryLen))
            .reshaped([queryLen, 1])
        let kIdx = MLXArray(Int32(0) ..< Int32(kvLen))
            .reshaped([1, kvLen])
        let dist = qIdx - kIdx
        let inside = MLX.logicalAnd(
            dist .> Int32(-window),
            dist .< Int32(window))

        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        let bias = MLX.where(inside, zero, negInf)
        // SDPA additive mask shape: broadcastable to (B, H, L, S). Add
        // leading singleton axes for batch and head.
        return .array(bias.reshaped([1, 1, queryLen, kvLen]))
    }
}
