// LastQueryPrefillV2.swift
//
// Final-layer prompt specialization for ContinuousBatchingV2.
//
// A prompt chunk's LAST decoder layer is asymmetric: every position's K/V
// must be committed to the cache (later tokens and decode attend them), but
// only the frontier position's ATTENTION OUTPUT is ever read — CBv2 samples
// from that one row and discards the rest (see PrefillOutputV2.swift).
//
// `updateAndAttendLastQuery` expresses exactly that: commit the full
// `[B, heads, L, D]` K/V rectangle, then evaluate attention for a single
// query row. Because the newest causal query can see every key the chunk
// just wrote, its result is bit-equivalent to the final row of ordinary
// chunk attention — no mask is needed, and a bound vision-span overlay
// cannot add future keys to that row either.
//
// Restrictions (enforced as preconditions in the implementation):
//  - FULL attention only. A sliding-window layer's queries have different
//    visible spans, so one query cannot stand in for the chunk.
//  - Storage-owning layers only (KV-shared layers borrow and write nothing).
//  - qL == 1 and kvL > 1.
//
// The cache still advances its per-row position offsets by the K/V length,
// NOT by the query length — the chunk really did consume `L` positions.

import Foundation
import MLX

/// Layer caches that can commit a full prompt chunk's K/V while attending
/// only its newest query. Conformance is a CAPABILITY claim: the contiguous
/// backend's `CBv2LayerCache` conforms; paged and custom caches do not, and
/// the model falls back to ordinary chunk attention for them.
public protocol CBv2LastQueryPrefillLayerCache: CBv2AttendingLayerCache {
    /// - queries: `[B, queryHeads, 1, headDim]` — the frontier row only,
    ///   already RoPE'd at the chunk's LAST absolute position.
    /// - keys/values: `[B, kvHeads, L, headDim]` — the COMPLETE chunk.
    /// Advances per-row offsets by `L` (the K/V length).
    func updateAndAttendLastQuery(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray
}
