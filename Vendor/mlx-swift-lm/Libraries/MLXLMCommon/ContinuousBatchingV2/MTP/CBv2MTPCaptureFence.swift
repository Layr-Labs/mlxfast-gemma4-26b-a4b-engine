// CBv2MTPCaptureFence.swift
//
// Ordering an MTP round's pre-write KV captures ahead of the in-place writes
// the very same graph is about to perform.

import MLX

/// Publishes a graph edge from a round's KV captures back into the storage
/// they were gathered from, so the round's own writes cannot overtake them.
///
/// Kept as a named seam rather than inlined in `mtpBuildVerifyGraph` for two
/// reasons: it is the one place in the MTP loop that knows a concrete storage
/// backend, and the edge it publishes is not observable from the round's
/// outputs, so it needs to be assertable on its own.
enum CBv2MTPCaptureFence {

    /// **Why this is not automatic.** On the contiguous backend `snapshot()`
    /// hands back the live `MLXArray`s and MLX's own versioning does the
    /// work: the round's `concatenated` update produces a *different* array,
    /// so the capture keeps referring to the pre-round bytes. On the PAGED
    /// backend it does not. `PagedSequenceKV.snapshot()` is a LAZY gather
    /// over slabs that the write kernels mutate IN PLACE, and in-place writes
    /// are invisible to MLX's hazard tracking. Both the capture and the
    /// round's writes merely *consume* the group's current `writeFence`
    /// (`PagedKVPool.gather`, `PagedKVPool.writeTokens`), so they are
    /// siblings in one unevaluated graph — first forced together at
    /// `executeMTPRound`'s `asyncEval` — and which one the scheduler runs
    /// first is an MLX detail, not a guarantee. If a write wins, the
    /// drafter's "frozen" pre-round KV already contains the round's own
    /// speculative tokens: drafts diverge from the target, acceptance
    /// collapses, and greedy token-exactness can break SILENTLY, with no
    /// crash to point at.
    ///
    /// It does not corrupt in practice today only because the windowed ring
    /// is still slightly wider than the widest range a row can gather. Do
    /// NOT read a token count out of this paragraph — this geometry has
    /// rotted three times. DERIVE it: `PagedKVPool.ringPageCount` sizes the
    /// ring as `ceil(max(PagedSequenceKV.maxWindowExposure(window) +
    /// CBv2PagedSpeculation.maxSpeculativeSpan, maxPrefillChunk) /
    /// pageSize)` pages, and the widest range an MTP round can ask the ring
    /// for is `maxWindowExposure(window)`. The margin is the difference
    /// between those two, in tokens.
    ///
    /// The WS-1.2/3.1 shrink HAS LANDED. Its first attempt aborted ordinary
    /// windowed prefill because the CACHE bound still carried a
    /// `maxPrefillChunk` term that a post-write gather needed;
    /// `PagedLayerCache` now assembles each chunk's KV BEFORE writing the
    /// chunk, the chunk term dropped out of that bound, and the ring shrank
    /// with it. At gemma-4's geometry (window 1,024, chunk 512, pageSize 16,
    /// span 8) the ring is 65 pages == 1,040 tokens against an attendable
    /// 1,024 — a margin of SIXTEEN tokens, where the pre-shrink 97-page ring
    /// left 528. Eight of those sixteen are the speculative span the formula
    /// reserves on purpose; the other eight are page rounding and would be
    /// gone at `window == 1,040`.
    ///
    /// So the slack this hazard used to hide behind is spent, and nothing
    /// asserts what remains. Read that as the fence being MORE load-bearing
    /// than the pre-shrink note claimed, not less.
    ///
    /// **This is why the edge is unconditional.** There IS a chain that ties
    /// ring sizing to speculation — `PagedKVPool.checkedRingPageCount` sizes
    /// against `CBv2PagedSpeculation.maxSpeculativeSpan`, and
    /// `assertSpanCoversMTPBound()` ties that span to the MTP draft bound —
    /// and this fence deliberately joins none of it. Gating the edge on ring
    /// geometry would make its correctness a function of three constants
    /// maintained in two other files, and would silently exempt the case
    /// with no ring at all. Publishing it unconditionally costs one element
    /// read per capture and is correct for any ring size, any span, any
    /// chunk, and for full-attention rows. It is a mechanism, not a check, so
    /// it cannot go stale when the ring geometry moves again — as it just
    /// did.
    ///
    /// **The fix: a fence BACK-edge.** `PagedKVPool.gather` already folds the
    /// group's write fence into its page index (`MLXArray(pages) +
    /// g.writeFence * 0`) so a read orders AFTER every prior write. That edge
    /// is one-directional. Here we run it the other way and fold the capture
    /// back INTO the fence, so every LATER write is forced to order after the
    /// gather: both write paths — `PagedKVPool.writeTokens`' bulk write and
    /// `PagedLayerCache`'s fused decode write — consume `group.writeFence`
    /// and re-publish its successor, so nothing reaches the slabs without
    /// first taking a dependency on the captures. `* 0` in `int32` is exactly
    /// zero for every input, including whatever an out-of-range float→int
    /// conversion produces, so the fence keeps its VALUE and gains only the
    /// graph edge. No host sync, so the round keeps its pipelining; the cost
    /// is ONE element of each capture, because MLX schedules whole
    /// primitives and a dependency on any slice forces the whole gather.
    ///
    /// A `.copy()` or a `stopGradient` would NOT do. The hazard is invisible
    /// to MLX, so the remedy has to be a real graph edge or a real
    /// evaluation — anything lazy just adds another sibling node.
    ///
    /// - Parameter captured: each capture paired with the row it was
    ///   gathered from.
    /// - Returns: the capture arrays that could NOT be fenced, because their
    ///   row exposes no write fence to hook. The caller must `eval` them:
    ///   that is the blunt equivalent, one host sync instead of a graph edge.
    ///   Empty for both backends that exist today — a future recyclable
    ///   backend must not silently inherit no protection at all.
    @discardableResult
    static func publish(
        _ captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)]
    ) -> [MLXArray] {
        var unfenceable: [MLXArray] = []
        for capture in captured {
            guard let paged = capture.row as? PagedSequenceKV else {
                unfenceable.append(capture.keys)
                unfenceable.append(capture.values)
                continue
            }
            // A zero-token capture (`PagedKVPool.gather` returns
            // `[1, H, 0, D]` for `count == 0`) has no element to probe. It
            // also has nothing to protect: that path never built a page
            // index, so it read no slab bytes and no later write can
            // clobber what it did not take. `.sum()` used to return 0 here
            // and publish an edge over nothing.
            guard capture.keys.dim(2) > 0, capture.values.dim(2) > 0 else { continue }
            let group = paged.pool.group(paged.groupKey)
            // ONE element of each is enough: MLX schedules whole
            // primitives, so a dependency on any slice of the gather forces
            // the gather itself. This used to be
            // `capture.keys.sum() + capture.values.sum()`, which publishes
            // the identical edge and reads the whole captured range to do
            // it — `PagedKVPool.gather` runs the opposite-direction edge
            // this way and names this site as the counterexample.
            let probe = capture.keys[0, 0, 0, 0] + capture.values[0, 0, 0, 0]
            group.writeFence = group.writeFence + probe.asType(group.writeFence.dtype) * 0
        }
        return unfenceable
    }
}
