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

public protocol CBv2LastQueryPrefillLayerCache: CBv2AttendingLayerCache {
    func updateAndAttendLastQuery(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray
}
